# Copyright 2018 Google LLC
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
"""Libraries for extending the Federated Language core library."""

# pylint: disable=g-importing-member
from federated_language.common_libs.tracing import propagate_trace_context_task_factory
from federated_language.common_libs.tracing import set_tracing_providers
from federated_language.common_libs.tracing import span
from federated_language.common_libs.tracing import trace
from federated_language.common_libs.tracing import TraceResult
from federated_language.common_libs.tracing import TracingProvider
from federated_language.common_libs.tracing import wrap_coroutine_in_current_trace_context
from federated_language.common_libs.tracing import wrap_rpc_in_trace_context
from federated_language.compiler.building_block_analysis import is_identity_function
from federated_language.compiler.building_block_factory import create_federated_aggregate
from federated_language.compiler.building_block_factory import create_federated_apply
from federated_language.compiler.building_block_factory import create_federated_broadcast
from federated_language.compiler.building_block_factory import create_federated_map
from federated_language.compiler.building_block_factory import create_federated_map_all_equal
from federated_language.compiler.building_block_factory import create_federated_map_or_apply
from federated_language.compiler.building_block_factory import create_federated_mean
from federated_language.compiler.building_block_factory import create_federated_secure_sum
from federated_language.compiler.building_block_factory import create_federated_secure_sum_bitwidth
from federated_language.compiler.building_block_factory import create_federated_select
from federated_language.compiler.building_block_factory import create_federated_sum
from federated_language.compiler.building_block_factory import create_federated_unzip
from federated_language.compiler.building_block_factory import create_federated_value
from federated_language.compiler.building_block_factory import create_federated_zip
from federated_language.compiler.building_block_factory import create_identity
from federated_language.compiler.building_block_factory import create_named_tuple
from federated_language.compiler.building_block_factory import create_sequence_map
from federated_language.compiler.building_block_factory import Path
from federated_language.compiler.building_block_factory import select_output_from_lambda
from federated_language.compiler.building_block_factory import unique_name_generator
from federated_language.compiler.building_blocks import Block
from federated_language.compiler.building_blocks import Call
from federated_language.compiler.building_blocks import CompiledComputation
from federated_language.compiler.building_blocks import ComputationBuildingBlock
from federated_language.compiler.building_blocks import Data
from federated_language.compiler.building_blocks import Intrinsic
from federated_language.compiler.building_blocks import Lambda
from federated_language.compiler.building_blocks import Literal
from federated_language.compiler.building_blocks import Placement
from federated_language.compiler.building_blocks import Reference
from federated_language.compiler.building_blocks import Selection
from federated_language.compiler.building_blocks import Struct
from federated_language.compiler.building_blocks import UnexpectedBlockError
from federated_language.compiler.intrinsic_defs import AggregationKind
from federated_language.compiler.intrinsic_defs import FEDERATED_AGGREGATE
from federated_language.compiler.intrinsic_defs import FEDERATED_APPLY
from federated_language.compiler.intrinsic_defs import FEDERATED_BROADCAST
from federated_language.compiler.intrinsic_defs import FEDERATED_EVAL_AT_CLIENTS
from federated_language.compiler.intrinsic_defs import FEDERATED_EVAL_AT_SERVER
from federated_language.compiler.intrinsic_defs import FEDERATED_MAP
from federated_language.compiler.intrinsic_defs import FEDERATED_MAP_ALL_EQUAL
from federated_language.compiler.intrinsic_defs import FEDERATED_MAX
from federated_language.compiler.intrinsic_defs import FEDERATED_MEAN
from federated_language.compiler.intrinsic_defs import FEDERATED_MIN
from federated_language.compiler.intrinsic_defs import FEDERATED_SECURE_SELECT
from federated_language.compiler.intrinsic_defs import FEDERATED_SECURE_SUM
from federated_language.compiler.intrinsic_defs import FEDERATED_SECURE_SUM_BITWIDTH
from federated_language.compiler.intrinsic_defs import FEDERATED_SELECT
from federated_language.compiler.intrinsic_defs import FEDERATED_SUM
from federated_language.compiler.intrinsic_defs import FEDERATED_VALUE_AT_CLIENTS
from federated_language.compiler.intrinsic_defs import FEDERATED_VALUE_AT_SERVER
from federated_language.compiler.intrinsic_defs import FEDERATED_WEIGHTED_MEAN
from federated_language.compiler.intrinsic_defs import FEDERATED_ZIP_AT_CLIENTS
from federated_language.compiler.intrinsic_defs import FEDERATED_ZIP_AT_SERVER
from federated_language.compiler.intrinsic_defs import GENERIC_DIVIDE
from federated_language.compiler.intrinsic_defs import GENERIC_MULTIPLY
from federated_language.compiler.intrinsic_defs import GENERIC_PLUS
from federated_language.compiler.intrinsic_defs import get_aggregation_intrinsics
from federated_language.compiler.intrinsic_defs import get_broadcast_intrinsics
from federated_language.compiler.intrinsic_defs import IntrinsicDef
from federated_language.compiler.intrinsic_defs import SEQUENCE_MAP
from federated_language.compiler.transformation_utils import BoundVariableTracker
from federated_language.compiler.transformation_utils import get_map_of_unbound_references
from federated_language.compiler.transformation_utils import SymbolTree
from federated_language.compiler.transformation_utils import transform_postorder
from federated_language.compiler.transformation_utils import transform_postorder_with_symbol_bindings
from federated_language.compiler.transformation_utils import transform_preorder
from federated_language.compiler.transformation_utils import TransformReturnType
from federated_language.compiler.transformation_utils import TransformSpec
from federated_language.compiler.tree_analysis import check_aggregate_not_dependent_on_aggregate
from federated_language.compiler.tree_analysis import check_broadcast_not_dependent_on_aggregate
from federated_language.compiler.tree_analysis import check_contains_no_new_unbound_references
from federated_language.compiler.tree_analysis import check_contains_no_unbound_references
from federated_language.compiler.tree_analysis import check_contains_only_reducible_intrinsics
from federated_language.compiler.tree_analysis import check_has_single_placement
from federated_language.compiler.tree_analysis import check_has_unique_names
from federated_language.compiler.tree_analysis import contains as computation_contains
from federated_language.compiler.tree_analysis import contains_called_intrinsic
from federated_language.compiler.tree_analysis import contains_no_unbound_references
from federated_language.compiler.tree_analysis import count as computation_count
from federated_language.compiler.tree_analysis import find_aggregations_in_tree
from federated_language.compiler.tree_analysis import NonuniqueNameError
from federated_language.compiler.tree_analysis import visit_postorder
from federated_language.compiler.tree_analysis import visit_preorder
from federated_language.computation.computation_base import Computation
from federated_language.computation.computation_impl import ConcreteComputation
from federated_language.computation.computation_wrapper import ComputationReturnedNoneError
from federated_language.computation.computation_wrapper import ComputationWrapper
from federated_language.computation.function_utils import pack_args_into_struct
from federated_language.computation.function_utils import unpack_arg
from federated_language.computation.function_utils import unpack_args_from_struct
from federated_language.computation.function_utils import wrap_as_zero_or_one_arg_callable
from federated_language.computation.polymorphic_computation import PolymorphicComputation
from federated_language.context_stack.context_base import AsyncContext
from federated_language.context_stack.context_base import ContextError
from federated_language.context_stack.context_base import SyncContext
from federated_language.context_stack.context_stack_base import ContextStack
from federated_language.context_stack.context_stack_test_utils import with_context
from federated_language.context_stack.context_stack_test_utils import with_contexts
from federated_language.context_stack.get_context_stack import get_context_stack
from federated_language.context_stack.runtime_error_context import RuntimeErrorContext
from federated_language.context_stack.set_default_context import set_default_context
from federated_language.context_stack.set_default_context import set_no_default_context
from federated_language.context_stack.symbol_binding_context import SymbolBindingContext
from federated_language.execution_contexts.async_execution_context import AsyncExecutionContext
from federated_language.execution_contexts.compiler_pipeline import CompilerPipeline
from federated_language.execution_contexts.sync_execution_context import SyncExecutionContext
from federated_language.executors.cardinalities_utils import infer_cardinalities
from federated_language.executors.executor_base import Executor
from federated_language.executors.executor_factory import CardinalitiesType
from federated_language.executors.executor_factory import ExecutorFactory
from federated_language.executors.executor_value_base import ExecutorValue
from federated_language.executors.executors_errors import RetryableError
from federated_language.federated_context.federated_computation_context import FederatedComputationContext
from federated_language.federated_context.value_utils import ensure_federated_value
from federated_language.test.static_assert import assert_contains_secure_aggregation
from federated_language.test.static_assert import assert_contains_unsecure_aggregation
from federated_language.test.static_assert import assert_not_contains_secure_aggregation
from federated_language.test.static_assert import assert_not_contains_unsecure_aggregation
from federated_language.types.computation_types import TypeNotAssignableError
from federated_language.types.computation_types import TypesNotEquivalentError
from federated_language.types.computation_types import UnexpectedTypeError
from federated_language.types.placements import PlacementLiteral
from federated_language.types.placements import uri_to_placement_literal
from federated_language.types.type_analysis import check_concrete_instance_of
from federated_language.types.type_analysis import check_is_structure_of_integers
from federated_language.types.type_analysis import contains as type_contains
from federated_language.types.type_analysis import contains_federated_types
from federated_language.types.type_analysis import contains_only as type_contains_only
from federated_language.types.type_analysis import contains_tensor_types
from federated_language.types.type_analysis import count as type_count
from federated_language.types.type_analysis import count_tensors_in_type
from federated_language.types.type_analysis import is_generic_op_compatible_type
from federated_language.types.type_analysis import is_single_integer_or_matches_structure
from federated_language.types.type_analysis import is_structure_of_floats
from federated_language.types.type_analysis import is_structure_of_integers
from federated_language.types.type_analysis import is_structure_of_tensors
from federated_language.types.type_analysis import preorder_types
from federated_language.types.type_conversions import infer_type
from federated_language.types.type_conversions import to_structure_with_type
from federated_language.types.type_conversions import type_to_py_container
from federated_language.types.type_test_utils import assert_type_assignable_from
from federated_language.types.type_test_utils import assert_types_equivalent
from federated_language.types.type_transformations import transform_type_postorder
# pylint: enable=g-importing-member
