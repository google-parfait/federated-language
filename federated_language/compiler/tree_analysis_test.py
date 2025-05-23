# Copyright 2019 Google LLC
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
from federated_language.compiler import building_block_factory
from federated_language.compiler import building_blocks
from federated_language.compiler import intrinsic_defs
from federated_language.compiler import tree_analysis
from federated_language.types import computation_types
from federated_language.types import placements
import numpy as np


def _create_test_federated_aggregate():
  value_type = computation_types.TensorType(np.int32)
  value = building_blocks.Literal(1, value_type)
  federated_value = building_block_factory.create_federated_value(
      value, placements.CLIENTS
  )
  accumulate_type = computation_types.StructType((value_type, value_type))
  accumulate = building_blocks.Lambda('a', accumulate_type, value)
  merge_type = computation_types.StructType((value_type, value_type))
  merge = building_blocks.Lambda('b', merge_type, value)
  report = building_blocks.Lambda('c', value_type, value)
  return building_block_factory.create_federated_aggregate(
      federated_value, value, accumulate, merge, report
  )


def _create_test_federated_broadcast():
  value_type = computation_types.TensorType(np.int32)
  value = building_blocks.Literal(1, value_type)
  federated_value = building_block_factory.create_federated_value(
      value, placements.SERVER
  )
  return building_block_factory.create_federated_broadcast(federated_value)


class TestCheckContainsOnlyReducibleIntrinsics(absltest.TestCase):

  def test_passes_with_federated_map(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri,
        computation_types.FunctionType(
            [
                computation_types.FunctionType(np.int32, np.float32),
                computation_types.FederatedType(np.int32, placements.CLIENTS),
            ],
            computation_types.FederatedType(np.float32, placements.CLIENTS),
        ),
    )
    tree_analysis.check_contains_only_reducible_intrinsics(intrinsic)

  def test_raises_with_federated_mean(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri,
        computation_types.FunctionType(
            computation_types.FederatedType(np.int32, placements.CLIENTS),
            computation_types.FederatedType(np.int32, placements.SERVER),
        ),
    )

    with self.assertRaisesRegex(ValueError, intrinsic.compact_representation()):
      tree_analysis.check_contains_only_reducible_intrinsics(intrinsic)


def whimsy_intrinsic_predicate(x):
  return (
      isinstance(x, building_blocks.Intrinsic) and x.uri == 'whimsy_intrinsic'
  )


class NodesDependentOnPredicateTest(absltest.TestCase):

  def test_raises_on_none_predicate(self):
    data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    with self.assertRaises(TypeError):
      tree_analysis._extract_nodes_consuming(data, None)

  def test_adds_all_nodes_to_set_with_constant_true_predicate(self):
    type_spec = computation_types.TensorType(np.int32)
    comp = building_blocks.Struct([
        building_blocks.Literal(1, type_spec),
        building_blocks.Literal(2, type_spec),
        building_blocks.Literal(3, type_spec),
    ])
    all_nodes = tree_analysis._extract_nodes_consuming(comp, lambda x: True)
    node_count = tree_analysis.count(comp)
    self.assertLen(all_nodes, node_count)

  def test_adds_no_nodes_to_set_with_constant_false_predicate(self):
    type_spec = computation_types.TensorType(np.int32)
    comp = building_blocks.Struct([
        building_blocks.Literal(1, type_spec),
        building_blocks.Literal(2, type_spec),
        building_blocks.Literal(3, type_spec),
    ])
    all_nodes = tree_analysis._extract_nodes_consuming(comp, lambda x: False)
    self.assertEmpty(all_nodes)

  def test_propogates_dependence_up_through_lambda(self):
    type_signature = computation_types.TensorType(np.int32)
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    lam = building_blocks.Lambda('x', np.int32, whimsy_intrinsic)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        lam, whimsy_intrinsic_predicate
    )
    self.assertIn(lam, dependent_nodes)

  def test_propogates_dependence_up_through_block_result(self):
    type_signature = computation_types.TensorType(np.int32)
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    integer_reference = building_blocks.Reference('int', np.int32)
    block = building_blocks.Block([('x', integer_reference)], whimsy_intrinsic)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        block, whimsy_intrinsic_predicate
    )
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_block_locals(self):
    type_signature = computation_types.TensorType(np.int32)
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    integer_reference = building_blocks.Reference('int', np.int32)
    block = building_blocks.Block([('x', whimsy_intrinsic)], integer_reference)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        block, whimsy_intrinsic_predicate
    )
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_tuple(self):
    type_signature = computation_types.TensorType(np.int32)
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    integer_reference = building_blocks.Reference('int', np.int32)
    tup = building_blocks.Struct([integer_reference, whimsy_intrinsic])
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        tup, whimsy_intrinsic_predicate
    )
    self.assertIn(tup, dependent_nodes)

  def test_propogates_dependence_up_through_selection(self):
    type_signature = computation_types.StructType([np.int32])
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    selection = building_blocks.Selection(whimsy_intrinsic, index=0)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        selection, whimsy_intrinsic_predicate
    )
    self.assertIn(selection, dependent_nodes)

  def test_propogates_dependence_up_through_call(self):
    type_signature = computation_types.TensorType(np.int32)
    whimsy_intrinsic = building_blocks.Intrinsic(
        'whimsy_intrinsic', type_signature
    )
    ref_to_x = building_blocks.Reference('x', np.int32)
    identity_lambda = building_blocks.Lambda('x', np.int32, ref_to_x)
    called_lambda = building_blocks.Call(identity_lambda, whimsy_intrinsic)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        called_lambda, whimsy_intrinsic_predicate
    )
    self.assertIn(called_lambda, dependent_nodes)

  def test_propogates_dependence_into_binding_to_reference(self):
    fed_type = computation_types.FederatedType(np.int32, placements.CLIENTS)
    ref_to_x = building_blocks.Reference('x', fed_type)
    federated_zero = building_blocks.Intrinsic(
        intrinsic_defs.GENERIC_ZERO.uri, fed_type
    )

    def federated_zero_predicate(x):
      return (
          isinstance(x, building_blocks.Intrinsic)
          and x.uri == intrinsic_defs.GENERIC_ZERO.uri
      )

    block = building_blocks.Block([('x', federated_zero)], ref_to_x)
    dependent_nodes = tree_analysis._extract_nodes_consuming(
        block, federated_zero_predicate
    )
    self.assertIn(ref_to_x, dependent_nodes)


class BroadcastDependentOnAggregateTest(absltest.TestCase):

  def test_does_not_find_aggregate_dependent_on_broadcast(self):
    broadcast = _create_test_federated_broadcast()
    value_type = broadcast.type_signature
    zero = building_blocks.Literal(1, value_type.member)
    accumulate_result = building_blocks.Literal(2, value_type.member)
    accumulate = building_blocks.Lambda(
        'accumulate_parameter',
        [value_type.member, value_type.member],
        accumulate_result,
    )
    merge_result = building_blocks.Literal(3, value_type.member)
    merge = building_blocks.Lambda(
        'merge_parameter', [value_type.member, value_type.member], merge_result
    )
    report_result = building_blocks.Literal(4, value_type.member)
    report = building_blocks.Lambda(
        'report_parameter', value_type.member, report_result
    )
    aggregate_dependent_on_broadcast = (
        building_block_factory.create_federated_aggregate(
            broadcast, zero, accumulate, merge, report
        )
    )
    tree_analysis.check_broadcast_not_dependent_on_aggregate(
        aggregate_dependent_on_broadcast
    )

  def test_finds_broadcast_dependent_on_aggregate(self):
    aggregate = _create_test_federated_aggregate()
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate
    )
    with self.assertRaises(ValueError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(
          broadcasted_aggregate
      )

  def test_returns_correct_example_of_broadcast_dependent_on_aggregate(self):
    aggregate = _create_test_federated_aggregate()
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate
    )
    with self.assertRaises(ValueError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(
          broadcasted_aggregate
      )


class AggregateDependentOnAggregateTest(absltest.TestCase):

  def test_does_not_find_aggregate_dependent_on_broadcast(self):
    broadcast = _create_test_federated_broadcast()
    value_type = broadcast.type_signature
    zero = building_blocks.Literal(1, value_type.member)
    accumulate_result = building_blocks.Literal(2, value_type.member)
    accumulate = building_blocks.Lambda(
        'accumulate_parameter',
        [value_type.member, value_type.member],
        accumulate_result,
    )
    merge_result = building_blocks.Literal(3, value_type.member)
    merge = building_blocks.Lambda(
        'merge_parameter', [value_type.member, value_type.member], merge_result
    )
    report_result = building_blocks.Literal(4, value_type.member)
    report = building_blocks.Lambda(
        'report_parameter', value_type.member, report_result
    )
    aggregate_dependent_on_broadcast = (
        building_block_factory.create_federated_aggregate(
            broadcast, zero, accumulate, merge, report
        )
    )
    tree_analysis.check_aggregate_not_dependent_on_aggregate(
        aggregate_dependent_on_broadcast
    )

  def test_finds_aggregate_dependent_on_aggregate(self):
    aggregate = _create_test_federated_aggregate()
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate
    )
    second_aggregate = building_block_factory.create_federated_sum(
        broadcasted_aggregate
    )
    with self.assertRaises(ValueError):
      tree_analysis.check_aggregate_not_dependent_on_aggregate(second_aggregate)


class ContainsCalledIntrinsic(absltest.TestCase):

  def test_returns_true_with_none_uri(self):
    comp = _create_test_federated_broadcast()
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp))

  def test_returns_true_with_matching_uri(self):
    comp = _create_test_federated_broadcast()
    uri = intrinsic_defs.FEDERATED_BROADCAST.uri
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp, uri))

  def test_returns_false_with_no_called_intrinsic(self):
    ref = building_blocks.Reference('a', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    self.assertFalse(tree_analysis.contains_called_intrinsic(fn))

  def test_returns_false_with_unmatched_called_intrinsic(self):
    comp = _create_test_federated_broadcast()
    uri = intrinsic_defs.FEDERATED_MAP.uri
    self.assertFalse(tree_analysis.contains_called_intrinsic(comp, uri))


class ContainsNoUnboundReferencesTest(absltest.TestCase):

  def test_raises_type_error_with_int_excluding(self):
    ref = building_blocks.Reference('a', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      tree_analysis.contains_no_unbound_references(fn, 1)

  def test_returns_true(self):
    ref = building_blocks.Reference('a', np.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    self.assertTrue(tree_analysis.contains_no_unbound_references(fn))

  def test_returns_true_with_excluded_reference(self):
    ref = building_blocks.Reference('a', np.int32)
    fn = building_blocks.Lambda('b', np.int32, ref)
    self.assertTrue(
        tree_analysis.contains_no_unbound_references(fn, excluding='a')
    )

  def test_returns_false(self):
    ref = building_blocks.Reference('a', np.int32)
    fn = building_blocks.Lambda('b', np.int32, ref)
    self.assertFalse(tree_analysis.contains_no_unbound_references(fn))


class CheckHasUniqueNamesTest(absltest.TestCase):

  def test_ok_on_single_lambda(self):
    ref_to_x = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, ref_to_x)
    tree_analysis.check_has_unique_names(lambda_1)

  def test_ok_on_multiple_no_arg_lambdas(self):
    data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    lambda_1 = building_blocks.Lambda(None, None, data)
    lambda_2 = building_blocks.Lambda(None, None, data)
    tup = building_blocks.Struct([lambda_1, lambda_2])
    tree_analysis.check_has_unique_names(tup)

  def test_raises_on_nested_lambdas_with_same_variable_name(self):
    ref_to_x = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, ref_to_x)
    lambda_2 = building_blocks.Lambda('x', np.int32, lambda_1)
    with self.assertRaises(tree_analysis.NonuniqueNameError):
      tree_analysis.check_has_unique_names(lambda_2)

  def test_ok_on_nested_lambdas_with_different_variable_name(self):
    ref_to_x = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, ref_to_x)
    lambda_2 = building_blocks.Lambda('y', np.int32, lambda_1)
    tree_analysis.check_has_unique_names(lambda_2)

  def test_ok_on_single_block(self):
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    single_block = building_blocks.Block([('x', x_data)], x_data)
    tree_analysis.check_has_unique_names(single_block)

  def test_raises_on_sequential_binding_of_same_variable_in_block(self):
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block([('x', x_data), ('x', x_data)], x_data)
    with self.assertRaises(tree_analysis.NonuniqueNameError):
      tree_analysis.check_has_unique_names(block)

  def test_ok_on_sequential_binding_of_different_variable_in_block(self):
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    block = building_blocks.Block([('x', x_data), ('y', x_data)], x_data)
    tree_analysis.check_has_unique_names(block)

  def test_raises_block_rebinding_of_lambda_variable(self):
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    single_block = building_blocks.Block([('x', x_data)], x_data)
    lambda_1 = building_blocks.Lambda('x', np.int32, single_block)
    with self.assertRaises(tree_analysis.NonuniqueNameError):
      tree_analysis.check_has_unique_names(lambda_1)

  def test_ok_block_binding_of_new_variable(self):
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    single_block = building_blocks.Block([('x', x_data)], x_data)
    lambda_1 = building_blocks.Lambda('y', np.int32, single_block)
    tree_analysis.check_has_unique_names(lambda_1)

  def test_raises_lambda_rebinding_of_block_variable(self):
    x_ref = building_blocks.Reference('x', np.int32)
    lambda_1 = building_blocks.Lambda('x', np.int32, x_ref)
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    single_block = building_blocks.Block([('x', x_data)], lambda_1)
    with self.assertRaises(tree_analysis.NonuniqueNameError):
      tree_analysis.check_has_unique_names(single_block)

  def test_ok_lambda_binding_of_new_variable(self):
    y_ref = building_blocks.Reference('y', np.int32)
    lambda_1 = building_blocks.Lambda('y', np.int32, y_ref)
    x_data = building_blocks.Literal(1, computation_types.TensorType(np.int32))
    single_block = building_blocks.Block([('x', x_data)], lambda_1)
    tree_analysis.check_has_unique_names(single_block)


if __name__ == '__main__':
  absltest.main()
