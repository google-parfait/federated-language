# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.context_stack import context
from federated_language.context_stack import context_stack_impl
from federated_language.program import federated_context
from federated_language.types import computation_types
from federated_language.types import placements
import numpy as np


class _TestContext(context.SyncContext):

  def invoke(self, comp, arg):
    raise AssertionError


class ContainsOnlyServerPlacedDataTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'struct_unnamed',
          computation_types.StructWithPythonType(
              [np.bool_, np.int32, np.str_], list
          ),
      ),
      (
          'struct_named',
          computation_types.StructWithPythonType(
              [
                  ('a', np.bool_),
                  ('b', np.int32),
                  ('c', np.str_),
              ],
              collections.OrderedDict,
          ),
      ),
      (
          'struct_nested',
          computation_types.StructWithPythonType(
              [
                  (
                      'x',
                      computation_types.StructWithPythonType(
                          [
                              ('a', np.bool_),
                              ('b', np.int32),
                          ],
                          collections.OrderedDict,
                      ),
                  ),
                  (
                      'y',
                      computation_types.StructWithPythonType(
                          [
                              ('c', np.str_),
                          ],
                          collections.OrderedDict,
                      ),
                  ),
              ],
              collections.OrderedDict,
          ),
      ),
      (
          'federated_struct',
          computation_types.FederatedType(
              computation_types.StructWithPythonType(
                  [
                      ('a', np.bool_),
                      ('b', np.int32),
                      ('c', np.str_),
                  ],
                  collections.OrderedDict,
              ),
              placements.SERVER,
          ),
      ),
      (
          'federated_sequence',
          computation_types.FederatedType(
              computation_types.SequenceType(np.int32), placements.SERVER
          ),
      ),
      (
          'federated_tensor',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
      ('sequence', computation_types.SequenceType(np.int32)),
      ('tensor', computation_types.TensorType(np.int32)),
  )
  def test_returns_true(self, type_signature):
    result = federated_context.contains_only_server_placed_data(type_signature)

    self.assertTrue(result)

  @parameterized.named_parameters(
      (
          'federated_tensor',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('placement', computation_types.PlacementType()),
  )
  def test_returns_false(self, type_signature):
    result = federated_context.contains_only_server_placed_data(type_signature)

    self.assertFalse(result)


class CheckInFederatedContextTest(parameterized.TestCase):

  def test_does_not_raise_value_error_with_context(self):
    mock_context = mock.create_autospec(
        federated_context.FederatedContext, spec_set=True, instance=True
    )

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    with context_stack_impl.context_stack.install(mock_context):
      try:
        federated_context.check_in_federated_context()
      except ValueError:
        self.fail('Raised `ValueError` unexpectedly.')

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

  def test_raises_value_error_with_context(self):
    mock_context = _TestContext()
    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    with context_stack_impl.context_stack.install(mock_context):
      with self.assertRaises(ValueError):
        federated_context.check_in_federated_context()

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

  def test_raises_value_error_with_context_nested(self):
    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    mock_context = mock.create_autospec(
        federated_context.FederatedContext, spec_set=True, instance=True
    )
    with context_stack_impl.context_stack.install(mock_context):
      try:
        federated_context.check_in_federated_context()
      except ValueError:
        self.fail('Raised `ValueError` unexpectedly.')

      mock_context = _TestContext()
      with context_stack_impl.context_stack.install(mock_context):
        with self.assertRaises(ValueError):
          federated_context.check_in_federated_context()

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()


if __name__ == '__main__':
  absltest.main()
