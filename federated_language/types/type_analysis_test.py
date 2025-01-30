# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
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
from federated_language.types import type_analysis
import ml_dtypes
import numpy as np


class CountTypesTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'one',
          computation_types.TensorType(np.int32),
          lambda t: isinstance(t, computation_types.TensorType),
          1,
      ),
      (
          'three',
          computation_types.StructType([np.int32] * 3),
          lambda t: isinstance(t, computation_types.TensorType),
          3,
      ),
      (
          'nested',
          computation_types.StructType([[np.int32] * 3] * 3),
          lambda t: isinstance(t, computation_types.TensorType),
          9,
      ),
  ])
  def test_returns_result(self, type_signature, predicate, expected_result):
    result = type_analysis.count(type_signature, predicate)
    self.assertEqual(result, expected_result)


class ContainsTypesTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'one_type',
          computation_types.TensorType(np.int32),
          computation_types.TensorType,
      ),
      (
          'two_types',
          computation_types.StructType([np.int32]),
          (computation_types.StructType, computation_types.TensorType),
      ),
      (
          'less_types',
          computation_types.TensorType(np.int32),
          (computation_types.StructType, computation_types.TensorType),
      ),
      (
          'more_types',
          computation_types.StructType([np.int32]),
          computation_types.TensorType,
      ),
  ])
  def test_returns_true(self, type_signature, types):
    result = type_analysis.contains(
        type_signature, lambda x: isinstance(x, types)
    )
    self.assertTrue(result)

  @parameterized.named_parameters([
      (
          'one_type',
          computation_types.TensorType(np.int32),
          computation_types.StructType,
      ),
  ])
  def test_returns_false(self, type_signature, types):
    result = type_analysis.contains(
        type_signature, lambda x: isinstance(x, types)
    )
    self.assertFalse(result)


class ContainsOnlyTypesTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'one_type',
          computation_types.TensorType(np.int32),
          computation_types.TensorType,
      ),
      (
          'two_types',
          computation_types.StructType([np.int32]),
          (computation_types.StructType, computation_types.TensorType),
      ),
      (
          'less_types',
          computation_types.TensorType(np.int32),
          (computation_types.StructType, computation_types.TensorType),
      ),
  ])
  def test_returns_true(self, type_signature, types):
    result = type_analysis.contains_only(
        type_signature, lambda x: isinstance(x, types)
    )
    self.assertTrue(result)

  @parameterized.named_parameters([
      (
          'one_type',
          computation_types.TensorType(np.int32),
          computation_types.StructType,
      ),
      (
          'more_types',
          computation_types.StructType([np.int32]),
          computation_types.TensorType,
      ),
  ])
  def test_returns_false(self, type_signature, types):
    result = type_analysis.contains_only(
        type_signature, lambda x: isinstance(x, types)
    )
    self.assertFalse(result)


class CheckIsSumCompatibleTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tensor_type', computation_types.TensorType(np.int32)),
      ('bfloat16_type', computation_types.TensorType(ml_dtypes.bfloat16)),
      (
          'struct_type_int',
          computation_types.StructType(
              [np.int32, np.int32],
          ),
      ),
      (
          'struct_type_float',
          computation_types.StructType([np.complex128, np.float32, np.float64]),
      ),
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
  ])
  def test_does_not_raise_sum_incompatible_error(self, type_spec):
    try:
      type_analysis.check_is_sum_compatible(type_spec)
    except type_analysis.SumIncompatibleError:
      self.fail('Raised `SumIncompatibleError` unexpectedly.')

  @parameterized.named_parameters([
      ('tensor_type_bool', computation_types.TensorType(np.bool_)),
      ('tensor_type_string', computation_types.TensorType(np.str_)),
      (
          'partially_defined_shape',
          computation_types.TensorType(np.int32, shape=[None]),
      ),
      ('tuple_type', computation_types.StructType([np.int32, np.bool_])),
      ('sequence_type', computation_types.SequenceType(np.int32)),
      ('placement_type', computation_types.PlacementType()),
      ('function_type', computation_types.FunctionType(np.int32, np.int32)),
      ('abstract_type', computation_types.AbstractType('T')),
  ])
  def test_raises_sum_incompatible_error(self, type_spec):
    with self.assertRaises(type_analysis.SumIncompatibleError):
      type_analysis.check_is_sum_compatible(type_spec)


class IsAverageCompatibleTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tensor_type_float32', computation_types.TensorType(np.float32)),
      ('tensor_type_float64', computation_types.TensorType(np.float64)),
      (
          'tuple_type',
          computation_types.StructType([('x', np.float32), ('y', np.float64)]),
      ),
      (
          'federated_type',
          computation_types.FederatedType(np.float32, placements.CLIENTS),
      ),
  ])
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_average_compatible(type_spec))

  @parameterized.named_parameters([
      ('tensor_type_int32', computation_types.TensorType(np.int32)),
      ('tensor_type_int64', computation_types.TensorType(np.int64)),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  ])
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_average_compatible(type_spec))


class IsMinMaxCompatibleTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tensor_type_int32', computation_types.TensorType(np.int32)),
      ('tensor_type_int64', computation_types.TensorType(np.int64)),
      ('tensor_type_float32', computation_types.TensorType(np.float32)),
      ('tensor_type_float64', computation_types.TensorType(np.float64)),
      (
          'tuple_type',
          computation_types.StructType([('x', np.float32), ('y', np.int64)]),
      ),
      (
          'federated_type',
          computation_types.FederatedType(np.float32, placements.CLIENTS),
      ),
  ])
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_min_max_compatible(type_spec))

  @parameterized.named_parameters([
      ('tensor_type_complex', computation_types.TensorType(np.complex128)),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  ])
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_min_max_compatible(type_spec))


class CheckFederatedTypeTest(absltest.TestCase):

  def test_passes_or_raises_type_error(self):
    type_spec = computation_types.FederatedType(
        np.int32, placements.CLIENTS, False
    )
    type_analysis.check_federated_type(
        type_spec,
        computation_types.TensorType(np.int32),
        placements.CLIENTS,
        False,
    )
    type_analysis.check_federated_type(
        type_spec, computation_types.TensorType(np.int32), None, None
    )
    type_analysis.check_federated_type(
        type_spec, None, placements.CLIENTS, None
    )
    type_analysis.check_federated_type(type_spec, None, None, False)
    self.assertRaises(
        TypeError,
        type_analysis.check_federated_type,
        type_spec,
        np.bool_,
        None,
        None,
    )
    self.assertRaises(
        TypeError,
        type_analysis.check_federated_type,
        type_spec,
        None,
        placements.SERVER,
        None,
    )
    self.assertRaises(
        TypeError,
        type_analysis.check_federated_type,
        type_spec,
        None,
        None,
        True,
    )


class IsStructureOfFloatsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_struct', computation_types.StructType([])),
      ('float', computation_types.TensorType(np.float32)),
      ('floats', computation_types.StructType([np.float32, np.float32])),
      (
          'nested_struct',
          computation_types.StructType([
              computation_types.TensorType(np.float32),
              computation_types.StructType([np.float32, np.float32]),
          ]),
      ),
      (
          'federated_float_at_clients',
          computation_types.FederatedType(np.float32, placements.CLIENTS),
      ),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_structure_of_floats(type_spec))

  @parameterized.named_parameters(
      ('bool', computation_types.TensorType(np.bool_)),
      ('int', computation_types.TensorType(np.int32)),
      ('string', computation_types.TensorType(np.str_)),
      ('float_and_bool', computation_types.StructType([np.float32, np.bool_])),
      (
          'nested_struct',
          computation_types.StructType([
              computation_types.TensorType(np.float32),
              computation_types.StructType([np.bool_, np.bool_]),
          ]),
      ),
      ('sequence_of_floats', computation_types.SequenceType(np.float32)),
      ('placement', computation_types.PlacementType()),
      ('function', computation_types.FunctionType(np.float32, np.float32)),
      ('abstract', computation_types.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_structure_of_floats(type_spec))


class IsStructureOfIntegersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_struct', computation_types.StructType([])),
      ('int', computation_types.TensorType(np.int32)),
      ('ints', computation_types.StructType([np.int32, np.int32])),
      (
          'nested_struct',
          computation_types.StructType([
              computation_types.TensorType(np.int32),
              computation_types.StructType([np.int32, np.int32]),
          ]),
      ),
      (
          'federated_int_at_clients',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(type_analysis.is_structure_of_integers(type_spec))

  @parameterized.named_parameters(
      ('bool', computation_types.TensorType(np.bool_)),
      ('float', computation_types.TensorType(np.float32)),
      ('string', computation_types.TensorType(np.str_)),
      ('int_and_bool', computation_types.StructType([np.int32, np.bool_])),
      (
          'nested_struct',
          computation_types.StructType([
              computation_types.TensorType(np.int32),
              computation_types.StructType([np.bool_, np.bool_]),
          ]),
      ),
      ('sequence_of_ints', computation_types.SequenceType(np.int32)),
      ('placement', computation_types.PlacementType()),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('abstract', computation_types.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(type_analysis.is_structure_of_integers(type_spec))


class IsSingleIntegerOrMatchesStructure(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'single int',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
      ),
      (
          'struct',
          computation_types.StructType([np.int32, np.int32]),
          computation_types.StructType([np.int32, np.int32]),
      ),
      (
          'struct with named fields',
          computation_types.StructType([('x', np.int32)]),
          computation_types.StructType([('x', np.int32)]),
      ),
      (
          'single int for complex tensor',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32, [5, 97, 204]),
      ),
      (
          'different kinds of ints',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int8),
      ),
      (
          'single int_for_struct',
          computation_types.TensorType(np.int32),
          computation_types.StructType([np.int32, np.int32]),
      ),
  )
  def test_returns_true(self, type_sig, shape_type):
    self.assertTrue(
        type_analysis.is_single_integer_or_matches_structure(
            type_sig, shape_type
        )
    )

  @parameterized.named_parameters(
      (
          'miscounted struct',
          computation_types.StructType([np.int32, np.int32, np.int32]),
          computation_types.StructType([np.int32, np.int32]),
      ),
      (
          'miscounted struct 2',
          computation_types.StructType([np.int32, np.int32]),
          computation_types.StructType([np.int32, np.int32, np.int32]),
      ),
      (
          'misnamed struct',
          computation_types.StructType([('x', np.int32)]),
          computation_types.StructType([('y', np.int32)]),
      ),
  )
  def test_returns_false(self, type_sig, shape_type):
    self.assertFalse(
        type_analysis.is_single_integer_or_matches_structure(
            type_sig, shape_type
        )
    )


class CheckConcreteInstanceOf(absltest.TestCase):

  def test_raises_with_int_first_argument(self):
    with self.assertRaises(TypeError):
      type_analysis.check_concrete_instance_of(
          1, computation_types.TensorType(np.int32)
      )

  def test_raises_with_int_second_argument(self):
    with self.assertRaises(TypeError):
      type_analysis.check_concrete_instance_of(
          computation_types.TensorType(np.int32), 1
      )

  def test_raises_different_structures(self):
    with self.assertRaises(type_analysis.MismatchedStructureError):
      type_analysis.check_concrete_instance_of(
          computation_types.TensorType(np.int32),
          computation_types.StructType([np.int32]),
      )

  def test_raises_with_abstract_type_as_first_arg(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(np.int32)
    with self.assertRaises(type_analysis.NotConcreteTypeError):
      type_analysis.check_concrete_instance_of(t1, t2)

  def test_with_single_abstract_type_and_tensor_type(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.TensorType(np.int32)
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_raises_with_abstract_type_in_first_and_second_argument(self):
    t1 = computation_types.AbstractType('T1')
    t2 = computation_types.AbstractType('T2')
    with self.assertRaises(type_analysis.NotConcreteTypeError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def func_with_param(self, param_type):
    return computation_types.FunctionType(
        param_type, computation_types.StructType([])
    )

  def test_with_single_abstract_type_and_tuple_type(self):
    t1 = self.func_with_param(computation_types.AbstractType('T1'))
    t2 = self.func_with_param(computation_types.StructType([np.int32]))
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_raises_with_conflicting_names(self):
    t1 = computation_types.StructType([np.int32] * 2)
    t2 = computation_types.StructType([('a', np.int32), ('b', np.int32)])
    with self.assertRaises(type_analysis.MismatchedStructureError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_raises_with_different_lengths(self):
    t1 = computation_types.StructType([np.int32] * 2)
    t2 = computation_types.StructType([np.int32])
    with self.assertRaises(type_analysis.MismatchedStructureError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_succeeds_under_tuple(self):
    t1 = self.func_with_param(
        computation_types.StructType([computation_types.AbstractType('T1')] * 2)
    )
    t2 = self.func_with_param(
        computation_types.StructType([
            computation_types.TensorType(np.int32),
            computation_types.TensorType(np.int32),
        ])
    )
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_fails_under_tuple_conflicting_concrete_types(self):
    t1 = self.func_with_param(
        computation_types.StructType([computation_types.AbstractType('T1')] * 2)
    )
    t2 = self.func_with_param(
        computation_types.StructType([
            computation_types.TensorType(np.int32),
            computation_types.TensorType(np.float32),
        ])
    )
    with self.assertRaises(type_analysis.MismatchedConcreteTypesError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_succeeds_abstract_type_under_sequence_type(self):
    t1 = self.func_with_param(
        computation_types.SequenceType(computation_types.AbstractType('T'))
    )
    t2 = self.func_with_param(computation_types.SequenceType(np.int32))
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_fails_conflicting_concrete_types_under_sequence(self):
    t1 = self.func_with_param(
        computation_types.SequenceType(
            [computation_types.AbstractType('T')] * 2
        )
    )
    t2 = self.func_with_param(
        computation_types.SequenceType([np.int32, np.float32])
    )
    with self.assertRaises(type_analysis.MismatchedConcreteTypesError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_succeeds_single_function_type(self):
    t1 = computation_types.FunctionType(
        *[computation_types.AbstractType('T')] * 2
    )
    t2 = computation_types.FunctionType(np.int32, np.int32)
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_succeeds_function_different_parameter_and_return_types(self):
    t1 = computation_types.FunctionType(
        computation_types.StructType([
            computation_types.AbstractType('U'),
            computation_types.AbstractType('T'),
        ]),
        computation_types.AbstractType('T'),
    )
    t2 = computation_types.FunctionType(
        computation_types.StructType([np.int32, np.float32]), np.float32
    )
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_fails_conflicting_binding_in_parameter_and_result(self):
    t1 = computation_types.FunctionType(
        computation_types.AbstractType('T'), computation_types.AbstractType('T')
    )
    t2 = computation_types.FunctionType(np.int32, np.float32)
    with self.assertRaises(type_analysis.UnassignableConcreteTypesError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_abstract_federated_types_succeeds(self):
    t1 = self.func_with_param(
        computation_types.FederatedType(
            [computation_types.AbstractType('T1')] * 2,
            placements.CLIENTS,
            all_equal=True,
        )
    )
    t2 = self.func_with_param(
        computation_types.FederatedType(
            [np.int32] * 2, placements.CLIENTS, all_equal=True
        )
    )
    type_analysis.check_concrete_instance_of(t2, t1)

  def test_abstract_fails_on_different_federated_placements(self):
    t1 = self.func_with_param(
        computation_types.FederatedType(
            [computation_types.AbstractType('T1')] * 2,
            placements.CLIENTS,
            all_equal=True,
        )
    )
    t2 = self.func_with_param(
        computation_types.FederatedType(
            [np.int32] * 2, placements.SERVER, all_equal=True
        )
    )
    with self.assertRaises(type_analysis.MismatchedStructureError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_abstract_can_be_concretized_fails_on_different_placements(self):
    t1 = self.func_with_param(
        computation_types.FederatedType(
            [computation_types.AbstractType('T1')] * 2,
            placements.CLIENTS,
            all_equal=True,
        )
    )
    t2 = self.func_with_param(
        computation_types.FederatedType(
            [np.int32] * 2, placements.SERVER, all_equal=True
        )
    )
    with self.assertRaises(type_analysis.MismatchedStructureError):
      type_analysis.check_concrete_instance_of(t2, t1)

  def test_abstract_parameters_contravariant(self):
    struct = lambda name: computation_types.StructType([(name, np.int32)])
    unnamed = struct(None)
    concrete = computation_types.FunctionType(
        computation_types.StructType(
            [unnamed, computation_types.FunctionType(struct('bar'), unnamed)]
        ),
        struct('foo'),
    )
    abstract = computation_types.AbstractType('A')
    generic = computation_types.FunctionType(
        computation_types.StructType(
            [abstract, computation_types.FunctionType(abstract, abstract)]
        ),
        abstract,
    )
    type_analysis.check_concrete_instance_of(concrete, generic)


if __name__ == '__main__':
  absltest.main()
