API Reference
=============

federated_language
------------------

*  federated_computation
*  federated_secure_sum
*  federated_secure_sum_bitwidth
*  to_value
*  program.FilteringReleaseManager
*  program.FederatedContext
*  program.contains_only_server_placed_data

.. automodule:: federated_language

.. autosummary::
  :toctree: generated

  Serializable
  Array
  array_from_proto
  array_from_proto_content
  array_is_compatible_dtype
  array_is_compatible_shape
  array_to_proto
  array_to_proto_content
  Computation
  federated_aggregate
  federated_broadcast
  federated_eval
  federated_map
  federated_max
  federated_mean
  federated_min
  federated_secure_select
  federated_secure_sum
  federated_secure_sum_bitwidth
  federated_select
  federated_sum
  federated_value
  federated_zip
  sequence_map
  sequence_reduce
  sequence_sum
  Value
  ArrayShape
  array_shape_from_proto
  array_shape_is_compatible_with
  array_shape_is_fully_defined
  array_shape_is_scalar
  num_elements_in_array_shape
  array_shape_to_proto
  AbstractType
  FederatedType
  FunctionType
  PlacementType
  SequenceType
  StructType
  StructWithPythonType
  TensorType
  to_type
  Type
  can_cast_dtype
  dtype_from_proto
  infer_dtype
  is_valid_dtype
  dtype_to_proto
  CLIENTS
  SERVER
  TypedObject

federated_language.program
--------------------------

.. automodule:: federated_language.program

.. autosummary::
  :toctree: generated

  FederatedDataSource
  FederatedDataSourceIterator
  check_in_federated_context
  ComputationArg
  LoggingReleaseManager
  MemoryReleaseManager
  ProgramStateExistsError
  ProgramStateManager
  ProgramStateNotFoundError
  ProgramStateStructure
  ProgramStateValue
  DelayedReleaseManager
  GroupingReleaseManager
  Key
  NotFilterableError
  PeriodicReleaseManager
  ReleasableStructure
  ReleasableValue
  ReleasedValueNotFoundError
  ReleaseManager
  MaterializableStructure
  MaterializableTypeSignature
  MaterializableValue
  MaterializableValueReference
  materialize_value
  MaterializedStructure
  MaterializedValue

federated_language.framework
----------------------------

.. automodule:: federated_language.framework

.. autosummary::
  :toctree: generated

  propagate_trace_context_task_factory
