API Reference
=============

:mod:`federated_language`
-------------------------

.. currentmodule:: federated_language

.. rubric:: Array

.. autosummary::

  Array
  array_from_proto
  array_from_proto_content
  array_to_proto
  array_to_proto_content
  array_is_compatible_dtype
  array_is_compatible_shape
  ArrayShape
  array_shape_from_proto
  array_shape_to_proto
  array_shape_is_fully_defined
  array_shape_is_scalar
  num_elements_in_array_shape
  dtype_from_proto
  dtype_to_proto
  can_cast_dtype
  is_valid_dtype
  infer_dtype

.. rubric:: Type

.. autosummary::

  Type
  AbstractType
  CLIENTS
  SERVER
  FederatedType
  FunctionType
  PlacementType
  SequenceType
  StructType
  StructWithPythonType
  TensorType
  to_type
  TypedObject

:mod:`federated_language.program`
---------------------------------

.. currentmodule:: federated_language.program

.. rubric:: Platform

.. autosummary::

  ComputationArg
  FederatedContext
  check_in_federated_context
  contains_only_server_placed_data
