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
# limitations under the License.
"""A library of transformation functions for computation types."""

from collections.abc import Callable

from federated_language.types import computation_types


def strip_placement(
    type_signature: computation_types.Type,
) -> computation_types.Type:
  """Removes instances of `FederatedType` from `type_signature`."""

  def _remove_placement(type_signature):
    if isinstance(type_signature, computation_types.FederatedType):
      return type_signature.member, True
    return type_signature, False

  return transform_type_postorder(type_signature, _remove_placement)[0]


# TODO: b/134525440 - Unifying the recursive methods in type_analysis.
def transform_type_postorder(
    type_signature: computation_types.Type,
    transform_fn: Callable[
        [computation_types.Type], tuple[computation_types.Type, bool]
    ],
) -> tuple[computation_types.Type, bool]:
  """Walks type tree of `type_signature` postorder, calling `transform_fn`.

  Args:
    type_signature: Instance of `computation_types.Type` to transform
      recursively.
    transform_fn: Transformation function to apply to each node in the type tree
      of `type_signature`. Must be instance of Python function type.

  Returns:
    A possibly transformed version of `type_signature`, with each node in its
    tree the result of applying `transform_fn` to the corresponding node in
    `type_signature`.

  Raises:
    NotImplementedError: If the types don't match the specification above.
  """
  if isinstance(type_signature, computation_types.FederatedType):
    transformed_member, member_mutated = transform_type_postorder(
        type_signature.member, transform_fn
    )
    if member_mutated:
      type_signature = computation_types.FederatedType(
          transformed_member, type_signature.placement, type_signature.all_equal
      )
    type_signature, type_signature_mutated = transform_fn(type_signature)
    return type_signature, type_signature_mutated or member_mutated
  elif isinstance(type_signature, computation_types.SequenceType):
    transformed_element, element_mutated = transform_type_postorder(
        type_signature.element, transform_fn
    )
    if element_mutated:
      type_signature = computation_types.SequenceType(transformed_element)
    type_signature, type_signature_mutated = transform_fn(type_signature)
    return type_signature, type_signature_mutated or element_mutated
  elif isinstance(type_signature, computation_types.FunctionType):
    if type_signature.parameter is not None:
      transformed_parameter, parameter_mutated = transform_type_postorder(
          type_signature.parameter, transform_fn
      )
    else:
      transformed_parameter, parameter_mutated = (None, False)
    transformed_result, result_mutated = transform_type_postorder(
        type_signature.result, transform_fn
    )
    if parameter_mutated or result_mutated:
      type_signature = computation_types.FunctionType(
          transformed_parameter, transformed_result
      )
    type_signature, type_signature_mutated = transform_fn(type_signature)
    return type_signature, (
        type_signature_mutated or parameter_mutated or result_mutated
    )
  elif isinstance(type_signature, computation_types.StructType):
    elements = []
    elements_mutated = False
    for element in type_signature.items():
      transformed_element, element_mutated = transform_type_postorder(
          element[1], transform_fn
      )
      elements_mutated = elements_mutated or element_mutated
      elements.append((element[0], transformed_element))
    if elements_mutated:
      if isinstance(type_signature, computation_types.StructWithPythonType):
        type_signature = computation_types.StructWithPythonType(
            elements,
            type_signature.python_container,
        )
      else:
        type_signature = computation_types.StructType(elements)
    type_signature, type_signature_mutated = transform_fn(type_signature)
    return type_signature, type_signature_mutated or elements_mutated
  elif isinstance(
      type_signature,
      (
          computation_types.AbstractType,
          computation_types.PlacementType,
          computation_types.TensorType,
      ),
  ):
    return transform_fn(type_signature)
  else:
    raise NotImplementedError(f'Unexpected type found: {type(type_signature)}.')
