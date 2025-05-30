# Copyright 2018 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_transformations
import numpy as np


def _convert_tensor_to_float(type_spec):
  if isinstance(type_spec, computation_types.TensorType):
    return computation_types.TensorType(np.float32, shape=type_spec.shape), True
  return type_spec, False


def _convert_abstract_type_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.AbstractType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


def _convert_placement_type_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.PlacementType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


def _convert_function_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.FunctionType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


def _convert_federated_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.FederatedType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


def _convert_sequence_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.SequenceType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


def _convert_tuple_to_tensor(type_spec):
  if isinstance(type_spec, computation_types.StructType):
    return computation_types.TensorType(np.float32), True
  return type_spec, False


class StripPlacementTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'noop_for_non_federated',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
      ),
      (
          'removes_server',
          computation_types.FederatedType(np.int32, placements.SERVER),
          computation_types.TensorType(np.int32),
      ),
      (
          'removes_clients',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.TensorType(np.int32),
      ),
      (
          'removes_nested',
          computation_types.StructType(
              [computation_types.FederatedType(np.int32, placements.SERVER)]
          ),
          computation_types.StructType([np.int32]),
      ),
      (
          'removes_multiple',
          computation_types.StructType([
              computation_types.FederatedType(np.int32, placements.SERVER),
              computation_types.FederatedType(np.float16, placements.CLIENTS),
          ]),
          computation_types.StructType([np.int32, np.float16]),
      ),
  ])
  def test_strips_placement(self, argument, expected):
    self.assertEqual(expected, type_transformations.strip_placement(argument))


class TransformTypePostorderTest(absltest.TestCase):

  def test_raises_on_none_function(self):
    with self.assertRaises(TypeError):
      type_transformations.transform_type_postorder(
          computation_types.TensorType(np.int32), None
      )

  def test_transforms_tensor(self):
    orig_type = computation_types.TensorType(np.int32)
    expected_type = computation_types.TensorType(np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_federated_type(self):
    orig_type = computation_types.FederatedType(np.int32, placements.CLIENTS)
    expected_type = computation_types.FederatedType(
        np.float32, placements.CLIENTS
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_federated_type(self):
    orig_type = computation_types.FederatedType([np.int32], placements.CLIENTS)
    expected_type = computation_types.FederatedType(
        [np.float32], placements.CLIENTS
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_federated(self):
    orig_type = computation_types.FederatedType(np.int32, placements.CLIENTS)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_federated_to_tensor
    )
    self.assertTrue(mutated)

  def test_transforms_sequence(self):
    orig_type = computation_types.SequenceType(np.int32)
    expected_type = computation_types.SequenceType(np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_sequence(self):
    orig_type = computation_types.SequenceType([np.int32])
    expected_type = computation_types.SequenceType([np.float32])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_sequence(self):
    orig_type = computation_types.SequenceType(np.int32)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_sequence_to_tensor
    )
    self.assertTrue(mutated)

  def test_transforms_function(self):
    orig_type = computation_types.FunctionType(np.int32, np.int64)
    expected_type = computation_types.FunctionType(np.float32, np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_function(self):
    orig_type = computation_types.FunctionType([np.int32], np.int64)
    expected_type = computation_types.FunctionType([np.float32], np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_function(self):
    orig_type = computation_types.FunctionType(np.int32, np.int64)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_function_to_tensor
    )
    self.assertTrue(mutated)

  def test_transforms_unnamed_tuple_type_preserving_tuple_container(self):
    orig_type = computation_types.StructWithPythonType(
        [np.int32, np.float64], tuple
    )
    expected_type = computation_types.StructWithPythonType(
        [np.float32, np.float32], tuple
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_unnamed_tuple_type(self):
    orig_type = computation_types.StructType([np.int32, np.float64])
    expected_type = computation_types.StructType([np.float32, np.float32])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_tuple(self):
    orig_type = computation_types.StructType([np.int32, np.float64])
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tuple_to_tensor
    )
    self.assertTrue(mutated)

  def test_transforms_named_tuple_type(self):
    orig_type = computation_types.StructType(
        [('a', np.int32), ('b', np.float64)]
    )
    expected_type = computation_types.StructType(
        [('a', np.float32), ('b', np.float32)]
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_named_tuple_type(self):
    orig_type = computation_types.StructType(
        [[('a', np.int32), ('b', np.float64)]]
    )
    expected_type = computation_types.StructType(
        [[('a', np.float32), ('b', np.float32)]]
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_named_tuple_type_preserving_tuple_container(self):
    orig_type = computation_types.StructWithPythonType(
        [('a', np.int32), ('b', np.float64)], dict
    )
    expected_type = computation_types.StructWithPythonType(
        [('a', np.float32), ('b', np.float32)], dict
    )
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_abstract_type(self):
    orig_type = computation_types.AbstractType('T')
    expected_type = computation_types.TensorType(np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_placement_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_placement_type(self):
    orig_type = computation_types.PlacementType()
    expected_type = computation_types.TensorType(np.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_placement_type_to_tensor
    )
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor
    )
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)


if __name__ == '__main__':
  absltest.main()
