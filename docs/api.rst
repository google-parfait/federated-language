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

:mod:`federated_language.framework`
===================================

.. automodule:: federated_language.framework
