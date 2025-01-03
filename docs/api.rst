API Reference
#############

:mod:`federated_language`
=========================

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

.. autoclass:: federated_language.Computation
.. autodecorator:: federated_language.federated_computation

.. autofunction:: federated_aggregate
.. autofunction:: federated_broadcast
.. autofunction:: federated_eval
.. autofunction:: federated_map
.. autofunction:: federated_max
.. autofunction:: federated_mean
.. autofunction:: federated_min
.. autofunction:: federated_secure_select
.. autofunction:: federated_select
.. autofunction:: federated_sum
.. autofunction:: federated_value
.. autofunction:: federated_zip
.. autofunction:: sequence_map
.. autofunction:: sequence_reduce
.. autofunction:: sequence_sum

.. autoclass:: federated_language.Value
.. autofunction:: federated_language.to_value

:mod:`federated_language`
=========================

.. automodule:: federated_language

.. autodata:: federated_language.Array
.. autodata:: federated_language.ArrayShape
.. autodata:: federated_language.CLIENTS
.. autodata:: federated_language.SERVER

:mod:`federated_language.program`
=================================

.. automodule:: federated_language.program

.. autodata:: federated_language.program.ComputationArg
.. autodata:: federated_language.program.Key
.. autodata:: federated_language.program.MaterializableStructure
.. autodata:: federated_language.program.MaterializableTypeSignature
.. autodata:: federated_language.program.MaterializableValue
.. autodata:: federated_language.program.MaterializedStructure
.. autodata:: federated_language.program.MaterializedValue
.. autodata:: federated_language.program.ProgramStateStructure
.. autodata:: federated_language.program.ProgramStateValue
.. autodata:: federated_language.program.ReleasableStructure
.. autodata:: federated_language.program.ReleasableValue

:mod:`federated_language.framework`
===================================

.. automodule:: federated_language.framework
