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
"""A library of static analysis functions for computation types."""

import collections
from collections.abc import Callable
from typing import Optional

from federated_language.types import array_shape
from federated_language.types import computation_types
from federated_language.types import placements
import ml_dtypes
import numpy as np

_TypePredicate = Callable[[computation_types.Type], bool]


def preorder_types(type_signature: computation_types.Type):
  """Yields each type in `type_signature` in a preorder fashion."""
  yield type_signature
  for child in type_signature.children():
    yield from preorder_types(child)


def count(
    type_signature: computation_types.Type, predicate: _TypePredicate
) -> int:
  """Returns the number of types in `type_signature` matching `predicate`.

  Args:
    type_signature: A tree of `computation_type.Type`s to count.
    predicate: A Python function that takes a type as a parameter and returns a
      boolean value.
  """
  one_or_zero = lambda t: 1 if predicate(t) else 0
  return sum(map(one_or_zero, preorder_types(type_signature)))


def contains(
    type_signature: computation_types.Type, predicate: _TypePredicate
) -> bool:
  """Checks if `type_signature` contains any types that pass `predicate`."""
  for t in preorder_types(type_signature):
    if predicate(t):
      return True
  return False


def contains_federated_types(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return contains(
      type_signature, lambda t: isinstance(t, computation_types.FederatedType)
  )


def contains_tensor_types(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return contains(
      type_signature, lambda t: isinstance(t, computation_types.TensorType)
  )


def contains_only(
    type_signature: computation_types.Type,
    predicate: _TypePredicate,
) -> bool:
  """Checks if `type_signature` contains only types that pass `predicate`."""
  return not contains(type_signature, lambda t: not predicate(t))


def is_structure_of_tensors(type_spec: computation_types.Type) -> bool:
  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.StructType,
            computation_types.TensorType,
        ),
    )

  return contains_only(type_spec, _predicate)


def is_generic_op_compatible_type(type_spec):
  """Checks `type_spec` against an explicit list of generic operators."""
  if type_spec is None:
    return False

  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.TensorType,
            computation_types.StructType,
        ),
    )

  return contains_only(type_spec, _predicate)


class SumIncompatibleError(TypeError):

  def __init__(self, type_spec, type_spec_context, reason):
    message = (
        'Expected a type which is compatible with the sum operator, found\n'
        f'{type_spec_context}\nwhich contains\n{type_spec}\nwhich is not '
        f'sum-compatible because {reason}.'
    )
    super().__init__(message)


def check_is_sum_compatible(
    type_spec: computation_types.Type,
    type_spec_context: Optional[computation_types.Type] = None,
):
  """Determines if `type_spec` is a type that can be added to itself.

  Types that are sum-compatible are composed of scalars of numeric types,
  possibly packaged into nested named tuples, and possibly federated. Types
  that are sum-incompatible include sequences, functions, abstract types,
  and placements.

  Args:
    type_spec: A `computation_types.Type`.
    type_spec_context: An optional parent type to include in the error message.

  Raises:
     SumIncompatibleError: if `type_spec` is not sum-compatible.
  """
  if type_spec_context is None:
    type_spec_context = type_spec
  if isinstance(type_spec, computation_types.TensorType):
    if not (
        np.issubdtype(type_spec.dtype, np.number)
        or type_spec.dtype == ml_dtypes.bfloat16
    ):
      raise SumIncompatibleError(
          type_spec, type_spec_context, f'{type_spec.dtype} is not numeric'
      )
    if not array_shape.is_shape_fully_defined(type_spec.shape):
      raise SumIncompatibleError(
          type_spec,
          type_spec_context,
          f'{type_spec.shape} is not fully defined',
      )
  elif isinstance(type_spec, computation_types.StructType):
    for _, element_type in type_spec.items():
      check_is_sum_compatible(element_type, type_spec_context)
  elif isinstance(type_spec, computation_types.FederatedType):
    check_is_sum_compatible(type_spec.member, type_spec_context)
  else:
    raise SumIncompatibleError(
        type_spec,
        type_spec_context,
        'only structures of tensors (possibly federated) may be summed',
    )


def is_structure_of_floats(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a structure of floats.

  Note that an empty `computation_types.StructType` will return `True`, as it
  does not contain any non-floating types.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is a structure of floats, otherwise `False`.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.floating)
  elif isinstance(type_spec, computation_types.StructType):
    return all(is_structure_of_floats(v) for _, v in type_spec.items())
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_structure_of_floats(type_spec.member)
  else:
    return False


def is_structure_of_integers(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is a structure of integers.

  Note that an empty `computation_types.StructType` will return `True`, as it
  does not contain any non-integer types.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is a structure of integers, otherwise `False`.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.integer)
  elif isinstance(type_spec, computation_types.StructType):
    return all(is_structure_of_integers(v) for _, v in type_spec.items())
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_structure_of_integers(type_spec.member)
  else:
    return False


def check_is_structure_of_integers(type_spec):
  if not is_structure_of_integers(type_spec):
    raise TypeError(
        'Expected a type which is structure of integers, found {}.'.format(
            type_spec
        )
    )


def is_single_integer_or_matches_structure(
    type_sig: computation_types.Type, shape_type: computation_types.Type
) -> bool:
  """If `type_sig` is an integer or integer structure matching `shape_type`."""
  if isinstance(type_sig, computation_types.TensorType):
    # This condition applies to both `shape_type` being a tensor or structure,
    # as the same integer bitwidth can be used for all values in the structure.
    return (
        np.issubdtype(type_sig.dtype, np.integer)
        and array_shape.num_elements_in_shape(type_sig.shape) == 1
    )
  elif isinstance(shape_type, computation_types.StructType) and isinstance(
      type_sig, computation_types.StructType
  ):
    bitwidth_name_and_types = list(type_sig.items())
    shape_name_and_types = list(shape_type.items())
    if len(type_sig) != len(shape_name_and_types):
      return False
    for (inner_name, type_sig), (inner_shape_name, inner_shape_type) in zip(
        bitwidth_name_and_types, shape_name_and_types
    ):
      if inner_name != inner_shape_name:
        return False
      if not is_single_integer_or_matches_structure(type_sig, inner_shape_type):
        return False
    return True
  else:
    return False


def check_federated_type(
    type_spec: computation_types.FederatedType,
    member: Optional[computation_types.Type] = None,
    placement: Optional[placements.PlacementLiteral] = None,
    all_equal: Optional[bool] = None,
):
  """Checks that `type_spec` is a federated type with the given parameters.

  Args:
    type_spec: The `federated_language.FederatedType` to check.
    member: The expected member type, or `None` if unspecified.
    placement: The desired placement, or `None` if unspecified.
    all_equal: The desired result of accessing the property
      `federated_language.FederatedType.all_equal` of `type_spec`, or `None` if
      left unspecified.

  Raises:
    TypeError: if `type_spec` is not a federated type of the given kind.
  """
  if member is not None:
    member.check_assignable_from(type_spec.member)
  if placement is not None:
    if type_spec.placement is not placement:
      raise TypeError(
          'Expected federated type placed at {}, got one placed at {}.'.format(
              placement, type_spec.placement
          )
      )
  if all_equal is not None:
    if type_spec.all_equal != all_equal:
      raise TypeError(
          'Expected federated type with all_equal {}, got one with {}.'.format(
              all_equal, type_spec.all_equal
          )
      )


def is_average_compatible(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` can be averaged.

  Types that are average-compatible are composed of numeric tensor types,
  either floating-point or complex, possibly packaged into nested named tuples,
  and possibly federated.

  Args:
    type_spec: a `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is average-compatible, `False` otherwise.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec, np.inexact)
  elif isinstance(type_spec, computation_types.StructType):
    return all(is_average_compatible(v) for _, v in type_spec.items())
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_average_compatible(type_spec.member)
  else:
    return False


def is_min_max_compatible(type_spec: computation_types.Type) -> bool:
  """Determines if `type_spec` is min/max compatible.

  Types that are min/max-compatible are composed of integer or floating tensor
  types, possibly packaged into nested tuples and possibly federated.

  Args:
    type_spec: a `computation_types.Type`.

  Returns:
    `True` iff `type_spec` is min/max compatible, `False` otherwise.
  """
  if isinstance(type_spec, computation_types.TensorType):
    return np.issubdtype(type_spec.dtype, np.integer) or np.issubdtype(
        type_spec.dtype, np.floating
    )
  elif isinstance(type_spec, computation_types.StructType):
    return all(is_min_max_compatible(v) for _, v in type_spec.items())
  elif isinstance(type_spec, computation_types.FederatedType):
    return is_min_max_compatible(type_spec.member)
  else:
    return False


class NotConcreteTypeError(TypeError):

  def __init__(self, full_type, found_abstract):
    message = (
        'Expected concrete type containing no abstract types, but '
        f'found abstract type {found_abstract} in {full_type}.'
    )
    super().__init__(message)


class MismatchedConcreteTypesError(TypeError):
  """Raised when there is a mismatch between two types."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      abstract_label,
      first_concrete,
      second_concrete,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but abstract type {abstract_label} '
        f'had substitutions {first_concrete} and {second_concrete}, which are '
        'not equivalent.'
    )
    super().__init__(message)


class UnassignableConcreteTypesError(TypeError):
  """Raised when one type can not be assigned to another type."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      abstract_label,
      definition,
      not_assignable_from,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but abstract type {abstract_label} '
        f'was defined as {definition}, and later used as {not_assignable_from} '
        'which cannot be assigned from the former.'
    )
    super().__init__(message)


class MismatchedStructureError(TypeError):
  """Raised when there is a mismatch between the structures of two types."""

  def __init__(
      self,
      full_concrete,
      full_generic,
      concrete_member,
      generic_member,
      mismatch,
  ):
    message = (
        f'Expected concrete type {full_concrete} to be a valid substitution '
        f'for generic type {full_generic}, but their structures do not match: '
        f'{concrete_member} differs in {mismatch} from {generic_member}.'
    )
    super().__init__(message)


class MissingDefiningUsageError(TypeError):

  def __init__(self, generic_type, label_name):
    message = (
        f'Missing defining use of abstract type {label_name} in type '
        f'{generic_type}. See `check_concrete_instance_of` documentation for '
        'details on what counts as a defining use.'
    )
    super().__init__(message)


def check_concrete_instance_of(
    concrete_type: computation_types.Type, generic_type: computation_types.Type
):
  """Checks whether `concrete_type` is a valid substitution of `generic_type`.

  This function determines whether `generic_type`'s type parameters can be
  substituted such that it is equivalent to `concrete type`.

  Note that passing through argument-position of function type swaps the
  variance of abstract types. Argument-position types can be assigned *from*
  other instances of the same type, but are not equivalent to it.

  Due to this variance issue, only abstract types must include at least one
  "defining" usage. "Defining" uses are those which are encased in function
  parameter position an odd number of times. These usages must all be
  equivalent. Non-defining usages need not compare equal but must be assignable
  *from* defining usages.

  Args:
    concrete_type: A type containing no `computation_types.AbstractType`s to
      check against `generic_type`'s shape.
    generic_type: A type which may contain `computation_types.AbstractType`s.

  Raises:
    TypeError: If `concrete_type` is not a valid substitution of `generic_type`.
  """
  for t in preorder_types(concrete_type):
    if isinstance(t, computation_types.AbstractType):
      raise NotConcreteTypeError(concrete_type, t)

  type_bindings = {}
  non_defining_usages = collections.defaultdict(list)

  def _check_helper(
      generic_type_member: computation_types.Type,
      concrete_type_member: computation_types.Type,
      defining: bool,
  ):
    """Recursive helper function."""

    def _raise_structural(mismatch):
      raise MismatchedStructureError(
          concrete_type,
          generic_type,
          concrete_type_member,
          generic_type_member,
          mismatch,
      )

    def _both_are(predicate):
      if predicate(generic_type_member):
        if predicate(concrete_type_member):
          return True
        else:
          _raise_structural('kind')
      else:
        return False

    if isinstance(generic_type_member, computation_types.AbstractType):
      label = str(generic_type_member.label)
      if not defining:
        non_defining_usages[label].append(concrete_type_member)
      else:
        bound_type = type_bindings.get(label)
        if bound_type is not None:
          if not concrete_type_member.is_equivalent_to(bound_type):
            raise MismatchedConcreteTypesError(
                concrete_type,
                generic_type,
                label,
                bound_type,
                concrete_type_member,
            )
        else:
          type_bindings[label] = concrete_type_member
    elif _both_are(lambda t: isinstance(t, computation_types.TensorType)):
      if generic_type_member != concrete_type_member:
        _raise_structural('tensor types')
    elif _both_are(lambda t: isinstance(t, computation_types.PlacementType)):
      if generic_type_member != concrete_type_member:
        _raise_structural('placements')
    elif _both_are(lambda t: isinstance(t, computation_types.StructType)):
      generic_elements = list(generic_type_member.items())  # pytype: disable=attribute-error
      concrete_elements = list(concrete_type_member.items())  # pytype: disable=attribute-error
      if len(generic_elements) != len(concrete_elements):
        _raise_structural('length')
      for generic_element, concrete_element in zip(
          generic_elements, concrete_elements
      ):
        if generic_element[0] != concrete_element[0]:
          _raise_structural('element names')
        _check_helper(generic_element[1], concrete_element[1], defining)
    elif _both_are(lambda t: isinstance(t, computation_types.SequenceType)):
      _check_helper(
          generic_type_member.element,  # pytype: disable=attribute-error
          concrete_type_member.element,  # pytype: disable=attribute-error
          defining,
      )
    elif _both_are(lambda t: isinstance(t, computation_types.FunctionType)):
      if generic_type_member.parameter is None:  # pytype: disable=attribute-error
        if concrete_type_member.parameter is not None:  # pytype: disable=attribute-error
          _raise_structural('parameter')
      else:
        _check_helper(
            generic_type_member.parameter,  # pytype: disable=attribute-error
            concrete_type_member.parameter,  # pytype: disable=attribute-error
            not defining,
        )
      _check_helper(
          generic_type_member.result,  # pytype: disable=attribute-error
          concrete_type_member.result,  # pytype: disable=attribute-error
          defining,
      )
    elif _both_are(lambda t: isinstance(t, computation_types.FederatedType)):
      if generic_type_member.placement != concrete_type_member.placement:  # pytype: disable=attribute-error
        _raise_structural('placement')
      if generic_type_member.all_equal != concrete_type_member.all_equal:  # pytype: disable=attribute-error
        _raise_structural('all equal')
      _check_helper(
          generic_type_member.member,  # pytype: disable=attribute-error
          concrete_type_member.member,  # pytype: disable=attribute-error
          defining,
      )
    else:
      raise TypeError(f'Unexpected type kind {generic_type}.')

  _check_helper(generic_type, concrete_type, False)

  for label, usages in non_defining_usages.items():
    bound_type = type_bindings.get(label)
    if bound_type is None:
      if len(usages) == 1:
        # Single-use abstract types can't be wrong.
        # NOTE: we could also add an exception here for cases where every usage
        # is equivalent to the first usage. However, that's not currently
        # needed since the only intrinsic that doesn't have a defining use is
        # GENERIC_ZERO, which has only a single-use type parameter.
        pass
      else:
        raise MissingDefiningUsageError(generic_type, label)
    else:
      for usage in usages:
        if not usage.is_assignable_from(bound_type):
          raise UnassignableConcreteTypesError(
              concrete_type, generic_type, label, bound_type, usage
          )
