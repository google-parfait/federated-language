# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.computation import computation_impl
from federated_language.executor import async_execution_context
from federated_language.federated_context import federated_computation
from federated_language.program import native_platform
from federated_language.program import program_test_utils
from federated_language.program import structure_utils
from federated_language.program import value_reference
from federated_language.types import computation_types
from federated_language.types import placements
import numpy as np
import tree


def _create_task(value: object) -> object:

  async def _fn(value: object) -> object:
    return value

  coro = _fn(value)
  return asyncio.create_task(coro)


def _create_identity_federated_computation(
    type_signature: computation_types.Type,
) -> computation_impl.ConcreteComputation:
  @federated_computation.federated_computation(type_signature)
  def _identity(value: object) -> object:
    return value

  return _identity


class NativeValueReferenceTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor_bool',
          lambda: _create_task(True),
          computation_types.TensorType(np.bool_),
          True,
      ),
      (
          'tensor_int',
          lambda: _create_task(1),
          computation_types.TensorType(np.int32),
          1,
      ),
      (
          'tensor_str',
          lambda: _create_task('abc'),
          computation_types.TensorType(np.str_),
          'abc',
      ),
      (
          'sequence',
          lambda: _create_task([1, 2, 3]),
          computation_types.SequenceType(np.int32),
          [1, 2, 3],
      ),
  )
  async def test_get_value_returns_value(
      self, task_factory, type_signature, expected_value
  ):
    task = task_factory()
    reference = native_platform.NativeValueReference(task, type_signature)

    actual_value = await reference.get_value()

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)


class CreateStructureOfReferencesTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor',
          lambda: _create_task(1),
          computation_types.TensorType(np.int32),
          lambda: native_platform.NativeValueReference(
              _create_task(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'sequence',
          lambda: _create_task([1, 2, 3]),
          computation_types.SequenceType(np.int32),
          lambda: native_platform.NativeValueReference(
              _create_task([1, 2, 3]), computation_types.SequenceType(np.int32)
          ),
      ),
      (
          'federated_server',
          lambda: _create_task(1),
          computation_types.FederatedType(np.int32, placements.SERVER),
          lambda: native_platform.NativeValueReference(
              _create_task(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'struct_unnamed',
          lambda: _create_task([True, 1, 'abc']),
          computation_types.StructWithPythonType(
              [np.bool_, np.int32, np.str_], list
          ),
          lambda: [
              native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
              native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
          ],
      ),
      (
          'struct_named_ordered',
          lambda: _create_task({'a': True, 'b': 1, 'c': 'abc'}),
          computation_types.StructWithPythonType(
              [
                  ('a', np.bool_),
                  ('b', np.int32),
                  ('c', np.str_),
              ],
              dict,
          ),
          lambda: {
              'a': native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
              'b': native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              'c': native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
          },
      ),
      (
          'struct_named_unordered',
          lambda: _create_task({'c': 'abc', 'b': 1, 'a': True}),
          computation_types.StructWithPythonType(
              [
                  ('c', np.str_),
                  ('b', np.int32),
                  ('a', np.bool_),
              ],
              dict,
          ),
          lambda: {
              'c': native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
              'b': native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              'a': native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
          },
      ),
      (
          'struct_nested',
          lambda: _create_task({'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}}),
          computation_types.StructWithPythonType(
              [
                  (
                      'x',
                      computation_types.StructWithPythonType(
                          [
                              ('a', np.bool_),
                              ('b', np.int32),
                          ],
                          dict,
                      ),
                  ),
                  (
                      'y',
                      computation_types.StructWithPythonType(
                          [
                              ('c', np.str_),
                          ],
                          dict,
                      ),
                  ),
              ],
              dict,
          ),
          lambda: {
              'x': {
                  'a': native_platform.NativeValueReference(
                      _create_task(True), computation_types.TensorType(np.bool_)
                  ),
                  'b': native_platform.NativeValueReference(
                      _create_task(1), computation_types.TensorType(np.int32)
                  ),
              },
              'y': {
                  'c': native_platform.NativeValueReference(
                      _create_task('abc'), computation_types.TensorType(np.str_)
                  ),
              },
          },
      ),
  )
  async def test_returns_value(
      self, task_factory, type_signature, expected_value_factory
  ):
    task = task_factory()
    actual_value = native_platform._create_structure_of_references(
        task, type_signature
    )

    expected_value = expected_value_factory()
    actual_value = await value_reference.materialize_value(actual_value)
    expected_value = await value_reference.materialize_value(expected_value)
    tree.assert_same_structure(actual_value, expected_value)
    program_test_utils.assert_same_key_order(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'federated_clients',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('placement', computation_types.PlacementType()),
  )
  async def test_raises_not_implemented_error_with_type_signature(
      self, type_signature
  ):
    task = _create_task(1)

    with self.assertRaises(NotImplementedError):
      native_platform._create_structure_of_references(task, type_signature)


class NativeFederatedContextTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor',
          _create_identity_federated_computation(
              computation_types.TensorType(np.int32)
          ),
          1,
          lambda: native_platform.NativeValueReference(
              _create_task(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'sequence',
          _create_identity_federated_computation(
              computation_types.SequenceType(np.int32)
          ),
          [1, 2, 3],
          lambda: native_platform.NativeValueReference(
              _create_task([1, 2, 3]), computation_types.SequenceType(np.int32)
          ),
      ),
      (
          'federated_server',
          _create_identity_federated_computation(
              computation_types.FederatedType(np.int32, placements.SERVER)
          ),
          1,
          lambda: native_platform.NativeValueReference(
              _create_task(1), computation_types.TensorType(np.int32)
          ),
      ),
      (
          'struct_unnamed',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [np.bool_, np.int32, np.str_], list
              )
          ),
          [True, 1, 'abc'],
          lambda: [
              native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
              native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
          ],
      ),
      (
          'struct_named_ordered',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      ('a', np.bool_),
                      ('b', np.int32),
                      ('c', np.str_),
                  ],
                  dict,
              )
          ),
          {'a': True, 'b': 1, 'c': 'abc'},
          lambda: {
              'a': native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
              'b': native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              'c': native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
          },
      ),
      (
          'struct_named_unordered',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      ('c', np.str_),
                      ('b', np.int32),
                      ('a', np.bool_),
                  ],
                  dict,
              )
          ),
          {'c': 'abc', 'b': 1, 'a': True},
          lambda: {
              'c': native_platform.NativeValueReference(
                  _create_task('abc'), computation_types.TensorType(np.str_)
              ),
              'b': native_platform.NativeValueReference(
                  _create_task(1), computation_types.TensorType(np.int32)
              ),
              'a': native_platform.NativeValueReference(
                  _create_task(True), computation_types.TensorType(np.bool_)
              ),
          },
      ),
      (
          'struct_nested',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          computation_types.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          lambda: {
              'x': {
                  'a': native_platform.NativeValueReference(
                      _create_task(True), computation_types.TensorType(np.bool_)
                  ),
                  'b': native_platform.NativeValueReference(
                      _create_task(1), computation_types.TensorType(np.int32)
                  ),
              },
              'y': {
                  'c': native_platform.NativeValueReference(
                      _create_task('abc'), computation_types.TensorType(np.str_)
                  ),
              },
          },
      ),
  )
  async def test_invoke_returns_result(
      self, comp, arg, expected_result_factory
  ):
    expected_result = expected_result_factory()
    expected_value = await value_reference.materialize_value(expected_result)
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    mock_context.invoke.return_value = expected_value
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)

    # Assert the result matches the expected structure of value references.
    tree.assert_same_structure(result, expected_result)
    program_test_utils.assert_same_key_order(result, expected_result)
    flattened = structure_utils.flatten(result)
    for element in flattened:
      self.assertIsInstance(element, native_platform.NativeValueReference)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      actual_value = await value_reference.materialize_value(result)

    # Assert the materialized value matches the expected structure of values.
    tree.assert_same_structure(actual_value, expected_value)
    program_test_utils.assert_same_key_order(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once_with(comp, arg)

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          computation_types.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_nested_partially_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', computation_types.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_sequentially(
      self, comp, arg, expected_value
  ):
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    mock_context.invoke.return_value = expected_value
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      flattened = structure_utils.flatten(result)
      materialized = [await v.get_value() for v in flattened]
      actual_value = structure_utils.unflatten_as(result, materialized)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once_with(comp, arg)

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          computation_types.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_nested_partially_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', computation_types.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_concurrently(
      self, comp, arg, expected_value
  ):
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    mock_context.invoke.return_value = expected_value
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await value_reference.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once_with(comp, arg)

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          computation_types.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_nested_partially_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      (
                          'x',
                          computation_types.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', computation_types.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_multiple(
      self, comp, arg, expected_value
  ):
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    mock_context.invoke.return_value = expected_value
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await asyncio.gather(
          value_reference.materialize_value(result),
          value_reference.materialize_value(result),
          value_reference.materialize_value(result),
      )

    expected_value = [expected_value] * 3
    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once_with(comp, arg)

  @parameterized.named_parameters(
      (
          'struct_unnamed_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType([], list)
          ),
          [],
          [],
      ),
      (
          'struct_named_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType([], dict)
          ),
          {},
          {},
      ),
      (
          'struct_nested_empty',
          _create_identity_federated_computation(
              computation_types.StructWithPythonType(
                  [
                      ('x', computation_types.StructWithPythonType([], dict)),
                      ('y', computation_types.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {}, 'y': {}},
          {'x': {}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_comp_not_called(
      self, comp, arg, expected_value
  ):
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await value_reference.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_not_called()

  @parameterized.named_parameters(
      (
          'federated_clients',
          _create_identity_federated_computation(
              computation_types.FederatedType(np.int32, placements.CLIENTS)
          ),
          1,
      ),
      (
          'function',
          _create_identity_federated_computation(
              computation_types.FunctionType(np.int32, np.int32)
          ),
          _create_identity_federated_computation(
              computation_types.TensorType(np.int32)
          ),
      ),
      (
          'placement',
          _create_identity_federated_computation(
              computation_types.PlacementType()
          ),
          None,
      ),
  )
  def test_invoke_raises_value_error_with_comp(self, comp, arg):
    mock_context = mock.create_autospec(
        async_execution_context.AsyncExecutionContext,
        spec_set=True,
        instance=True,
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with self.assertRaises(ValueError):
      context.invoke(comp, arg)


if __name__ == '__main__':
  absltest.main()
