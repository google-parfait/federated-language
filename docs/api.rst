API Reference
#############

:mod:`federated_language`
=========================

..
  Array
.. autodata:: federated_language.Array
.. autofunction:: federated_language.array_from_proto
.. autofunction:: federated_language.array_from_proto_content
.. autofunction:: federated_language.array_to_proto
.. autofunction:: federated_language.array_to_proto_content
.. autofunction:: federated_language.array_is_compatible_dtype
.. autofunction:: federated_language.array_is_compatible_shape
.. autodata:: federated_language.ArrayShape
.. autofunction:: federated_language.array_shape_from_proto
.. autofunction:: federated_language.array_shape_to_proto
.. autofunction:: federated_language.array_shape_is_fully_defined
.. autofunction:: federated_language.array_shape_is_scalar
.. autofunction:: federated_language.num_elements_in_array_shape
.. autofunction:: federated_language.dtype_from_proto
.. autofunction:: federated_language.dtype_to_proto
.. autofunction:: federated_language.can_cast_dtype
.. autofunction:: federated_language.is_valid_dtype
.. autofunction:: federated_language.infer_dtype

..
  Type
.. autoclass:: federated_language.Type
.. autoclass:: federated_language.AbstractType
.. autodata:: federated_language.CLIENTS
.. autodata:: federated_language.SERVER
.. autoclass:: federated_language.FederatedType
.. autoclass:: federated_language.FunctionType
.. autoclass:: federated_language.PlacementType
.. autoclass:: federated_language.SequenceType
.. autoclass:: federated_language.StructType
.. autoclass:: federated_language.StructWithPythonType
.. autoclass:: federated_language.TensorType
.. autofunction:: federated_language.to_type
.. autoclass:: federated_language.TypedObject

..
  Computation
.. autoclass:: federated_language.Computation
.. autodecorator:: federated_language.federated_computation

..
  Intrinsics
.. autofunction:: federated_language.federated_aggregate
.. autofunction:: federated_language.federated_broadcast
.. autofunction:: federated_language.federated_eval
.. autofunction:: federated_language.federated_map
.. autofunction:: federated_language.federated_max
.. autofunction:: federated_language.federated_mean
.. autofunction:: federated_language.federated_min
.. autofunction:: federated_language.federated_secure_select
.. autofunction:: federated_language.federated_select
.. autofunction:: federated_language.federated_sum
.. autofunction:: federated_language.federated_value
.. autofunction:: federated_language.federated_zip
.. autofunction:: federated_language.sequence_map
.. autofunction:: federated_language.sequence_reduce
.. autofunction:: federated_language.sequence_sum

..
  Other
.. autoclass:: federated_language.Value
.. autofunction:: federated_language.to_value

:mod:`federated_language.program`
=================================

..
  Platform
.. autodata:: federated_language.program.ComputationArg
.. autoclass:: federated_language.program.FederatedContext
.. autofunction:: federated_language.program.check_in_federated_context
.. autofunction:: federated_language.program.contains_only_server_placed_data

..
  Value Reference
.. autodata:: federated_language.program.MaterializableTypeSignature
.. autodata:: federated_language.program.MaterializableValue
.. autodata:: federated_language.program.MaterializableStructure
.. autoclass:: federated_language.program.MaterializableValueReference
.. autofunction:: federated_language.program.materialize_value
.. autodata:: federated_language.program.MaterializedValue
.. autodata:: federated_language.program.MaterializedStructure

..
  Data
.. autoclass:: federated_language.program.FederatedDataSource
.. autoclass:: federated_language.program.FederatedDataSourceIterator

..
  Release
.. autodata:: federated_language.program.ReleasableValue
.. autodata:: federated_language.program.ReleasableStructure
.. autodata:: federated_language.program.Key
.. autoclass:: federated_language.program.ReleaseManager
.. autoclass:: federated_language.program.FilteringReleaseManager
.. autoclass:: federated_language.program.GroupingReleaseManager
.. autoclass:: federated_language.program.PeriodicReleaseManager
.. autoclass:: federated_language.program.DelayedReleaseManager
.. autoclass:: federated_language.program.LoggingReleaseManager
.. autoclass:: federated_language.program.MemoryReleaseManager
.. autoerror:: federated_language.program.NotFilterableError
.. autoerror:: federated_language.program.ReleasedValueNotFoundError

..
  Fault Tolerance
.. autodata:: federated_language.program.ProgramStateValue
.. autodata:: federated_language.program.ProgramStateStructure
.. autoclass:: federated_language.program.ProgramStateManager
.. autoerror:: federated_language.program.ProgramStateExistsError
.. autoerror:: federated_language.program.ProgramStateNotFoundError

:mod:`federated_language.framework`
===================================

..
  Type
.. autoerror:: federated_language.framework.TypeNotAssignableError
.. autoerror:: federated_language.framework.TypesNotEquivalentError
.. autoerror:: federated_language.framework.TypesNotIdenticalError
.. autoerror:: federated_language.framework.UnexpectedTypeError
.. autoclass:: federated_language.framework.PlacementLiteral
.. autofunction:: federated_language.framework.uri_to_placement_literal
.. autofunction:: federated_language.framework.check_concrete_instance_of
.. autofunction:: federated_language.framework.check_is_structure_of_integers
.. autofunction:: federated_language.framework.contains as type_contains
.. autofunction:: federated_language.framework.contains_federated_types
.. autofunction:: federated_language.framework.contains_only as type_contains_only
.. autofunction:: federated_language.framework.contains_tensor_types
.. autofunction:: federated_language.framework.count as type_count
.. autofunction:: federated_language.framework.count_tensors_in_type
.. autofunction:: federated_language.framework.is_generic_op_compatible_type
.. autofunction:: federated_language.framework.is_single_integer_or_matches_structure
.. autofunction:: federated_language.framework.is_structure_of_floats
.. autofunction:: federated_language.framework.is_structure_of_integers
.. autofunction:: federated_language.framework.is_structure_of_tensors
.. autofunction:: federated_language.framework.preorder_types
.. autofunction:: federated_language.framework.infer_type
.. autofunction:: federated_language.framework.to_structure_with_type
.. autofunction:: federated_language.framework.type_to_non_all_equal
.. autofunction:: federated_language.framework.type_to_py_container
.. autofunction:: federated_language.framework.assert_type_assignable_from
.. autofunction:: federated_language.framework.assert_types_equivalent
.. autofunction:: federated_language.framework.assert_types_identical
.. autofunction:: federated_language.framework.transform_type_postorder

..
  Computation
.. autoclass:: federated_language.framework.Computation
.. autoclass:: federated_language.framework.ConcreteComputation
.. autoerror:: federated_language.framework.ComputationReturnedNoneError
.. autoclass:: federated_language.framework.ComputationWrapper
.. autofunction:: federated_language.framework.pack_args_into_struct
.. autofunction:: federated_language.framework.unpack_arg
.. autofunction:: federated_language.framework.unpack_args_from_struct
.. autofunction:: federated_language.framework.wrap_as_zero_or_one_arg_callable
.. autoclass:: federated_language.framework.PolymorphicComputation

..
  Context
.. autoclass:: federated_language.framework.AsyncContext
.. autoerror:: federated_language.framework.ContextError
.. autoclass:: federated_language.framework.SyncContext
.. autoclass:: federated_language.framework.ContextStack
.. autofunction:: federated_language.framework.global_context_stack
.. autofunction:: federated_language.framework.with_context
.. autofunction:: federated_language.framework.with_contexts
.. autofunction:: federated_language.framework.get_context_stack
.. autofunction:: federated_language.framework.create_runtime_error_context
.. autoclass:: federated_language.framework.RuntimeErrorContext
.. autofunction:: federated_language.framework.set_default_context
.. autofunction:: federated_language.framework.set_no_default_context
.. autoclass:: federated_language.framework.SymbolBindingContext

..
  Intrinsics
.. autoclass:: federated_language.framework.FederatedComputationContext
.. autofunction:: federated_language.framework.ensure_federated_value

..
  Compiler
.. autofunction:: federated_language.framework.is_identity_function
.. autofunction:: federated_language.framework.create_federated_aggregate
.. autofunction:: federated_language.framework.create_federated_apply
.. autofunction:: federated_language.framework.create_federated_broadcast
.. autofunction:: federated_language.framework.create_federated_map
.. autofunction:: federated_language.framework.create_federated_map_all_equal
.. autofunction:: federated_language.framework.create_federated_map_or_apply
.. autofunction:: federated_language.framework.create_federated_mean
.. autofunction:: federated_language.framework.create_federated_secure_sum
.. autofunction:: federated_language.framework.create_federated_secure_sum_bitwidth
.. autofunction:: federated_language.framework.create_federated_select
.. autofunction:: federated_language.framework.create_federated_sum
.. autofunction:: federated_language.framework.create_federated_unzip
.. autofunction:: federated_language.framework.create_federated_value
.. autofunction:: federated_language.framework.create_federated_zip
.. autofunction:: federated_language.framework.create_identity
.. autofunction:: federated_language.framework.create_named_tuple
.. autofunction:: federated_language.framework.create_sequence_map
.. autodata:: federated_language.framework.Path
.. autofunction:: federated_language.framework.select_output_from_lambda
.. autofunction:: federated_language.framework.unique_name_generator
.. autoclass:: federated_language.framework.Block
.. autoclass:: federated_language.framework.Call
.. autoclass:: federated_language.framework.CompiledComputation
.. autoclass:: federated_language.framework.ComputationBuildingBlock
.. autoclass:: federated_language.framework.Data
.. autoclass:: federated_language.framework.Intrinsic
.. autoclass:: federated_language.framework.Lambda
.. autoclass:: federated_language.framework.Literal
.. autoclass:: federated_language.framework.Placement
.. autoclass:: federated_language.framework.Reference
.. autoclass:: federated_language.framework.Selection
.. autoclass:: federated_language.framework.Struct
.. autoerror:: federated_language.framework.UnexpectedBlockError
.. autodata:: federated_language.framework.AggregationKind
.. autodata:: federated_language.framework.FEDERATED_AGGREGATE
.. autodata:: federated_language.framework.FEDERATED_APPLY
.. autodata:: federated_language.framework.FEDERATED_BROADCAST
.. autodata:: federated_language.framework.FEDERATED_EVAL_AT_CLIENTS
.. autodata:: federated_language.framework.FEDERATED_EVAL_AT_SERVER
.. autodata:: federated_language.framework.FEDERATED_MAP
.. autodata:: federated_language.framework.FEDERATED_MAP_ALL_EQUAL
.. autodata:: federated_language.framework.FEDERATED_MAX
.. autodata:: federated_language.framework.FEDERATED_MEAN
.. autodata:: federated_language.framework.FEDERATED_MIN
.. autodata:: federated_language.framework.FEDERATED_SECURE_SELECT
.. autodata:: federated_language.framework.FEDERATED_SECURE_SUM
.. autodata:: federated_language.framework.FEDERATED_SECURE_SUM_BITWIDTH
.. autodata:: federated_language.framework.FEDERATED_SELECT
.. autodata:: federated_language.framework.FEDERATED_SUM
.. autodata:: federated_language.framework.FEDERATED_VALUE_AT_CLIENTS
.. autodata:: federated_language.framework.FEDERATED_VALUE_AT_SERVER
.. autodata:: federated_language.framework.FEDERATED_WEIGHTED_MEAN
.. autodata:: federated_language.framework.FEDERATED_ZIP_AT_CLIENTS
.. autodata:: federated_language.framework.FEDERATED_ZIP_AT_SERVER
.. autodata:: federated_language.framework.GENERIC_DIVIDE
.. autodata:: federated_language.framework.GENERIC_MULTIPLY
.. autodata:: federated_language.framework.GENERIC_PLUS
.. autodata:: federated_language.framework.get_aggregation_intrinsics
.. autodata:: federated_language.framework.get_broadcast_intrinsics
.. autodata:: federated_language.framework.IntrinsicDef
.. autodata:: federated_language.framework.SEQUENCE_MAP
.. autoclass:: federated_language.framework.BoundVariableTracker
.. autofunction:: federated_language.framework.get_map_of_unbound_references
.. autoclass:: federated_language.framework.SymbolTree
.. autofunction:: federated_language.framework.transform_postorder
.. autofunction:: federated_language.framework.transform_postorder_with_symbol_bindings
.. autofunction:: federated_language.framework.transform_preorder
.. autodata:: federated_language.framework.TransformReturnType
.. autoclass:: federated_language.framework.TransformSpec
.. autofunction:: federated_language.framework.check_aggregate_not_dependent_on_aggregate
.. autofunction:: federated_language.framework.check_broadcast_not_dependent_on_aggregate
.. autofunction:: federated_language.framework.check_contains_no_new_unbound_references
.. autofunction:: federated_language.framework.check_contains_no_unbound_references
.. autofunction:: federated_language.framework.check_contains_only_reducible_intrinsics
.. autofunction:: federated_language.framework.check_has_single_placement
.. autofunction:: federated_language.framework.check_has_unique_names
.. autofunction:: federated_language.framework.contains as computation_contains
.. autofunction:: federated_language.framework.contains_called_intrinsic
.. autofunction:: federated_language.framework.contains_no_unbound_references
.. autofunction:: federated_language.framework.count as computation_count
.. autofunction:: federated_language.framework.find_aggregations_in_tree
.. autoerror:: federated_language.framework.NonuniqueNameError
.. autofunction:: federated_language.framework.visit_postorder
.. autofunction:: federated_language.framework.visit_preorder
.. autofunction:: federated_language.framework.
..
  Executor
.. autoclass:: federated_language.framework.AsyncExecutionContext
.. autoclass:: federated_language.framework.CompilerPipeline
.. autoclass:: federated_language.framework.SyncExecutionContext
.. autofunction:: federated_language.framework.infer_cardinalities
.. autoclass:: federated_language.framework.Executor
.. autodata:: federated_language.framework.CardinalitiesType
.. autoclass:: federated_language.framework.ExecutorFactory
.. autoclass:: federated_language.framework.ExecutorValue
.. autoerror:: federated_language.framework.RetryableError

..
  Analysis
.. autofunction:: federated_language.framework.assert_contains_secure_aggregation
.. autofunction:: federated_language.framework.assert_contains_unsecure_aggregation
.. autofunction:: federated_language.framework.assert_not_contains_secure_aggregation
.. autofunction:: federated_language.framework.assert_not_contains_unsecure_aggregation

..
  Other
.. autofunction:: federated_language.framework.propagate_trace_context_task_factory
.. autofunction:: federated_language.framework.set_tracing_providers
.. autofunction:: federated_language.framework.span
.. autofunction:: federated_language.framework.trace
.. autoclass:: federated_language.framework.TraceResult
.. autoclass:: federated_language.framework.TracingProvider
.. autofunction:: federated_language.framework.wrap_coroutine_in_current_trace_context
.. autofunction:: federated_language.framework.wrap_rpc_in_trace_context
