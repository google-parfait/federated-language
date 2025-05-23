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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities file for functions with `Value`s as inputs and outputs."""

from typing import Optional

from federated_language.compiler import building_block_factory
from federated_language.compiler import building_blocks
from federated_language.federated_context import value_impl
from federated_language.types import computation_types
from federated_language.types import placements


def get_curried(fn: value_impl.Value):
  """Returns a curried version of function `fn` that takes a parameter tuple.

  For functions `fn` of types <T1,T2,....,Tn> -> U, the result is a function
  of the form T1 -> (T2 -> (T3 -> .... (Tn -> U) ... )).

  NOTE: No attempt is made at avoiding naming conflicts in cases where `fn`
  contains references. The arguments of the curriend function are named `argN`
  with `N` starting at 0.

  Args:
    fn: A value of a functional type.

  Returns:
    A value that represents the curried form of `fn`.
  """
  if not isinstance(fn.type_signature, computation_types.FunctionType):
    raise ValueError(
        'Expected a `federated_language.FunctionType`, found'
        f' {fn.type_signature}.'
    )
  if not isinstance(fn.type_signature.parameter, computation_types.StructType):
    raise ValueError(
        'Expected a `federated_language.StructType`, found'
        f' {fn.type_signature.parameter}.'
    )
  param_elements = fn.type_signature.parameter.items()
  references = []
  for idx, (_, elem_type) in enumerate(param_elements):
    references.append(building_blocks.Reference('arg{}'.format(idx), elem_type))
  result = building_blocks.Call(fn.comp, building_blocks.Struct(references))
  for ref in references[::-1]:
    result = building_blocks.Lambda(ref.name, ref.type_signature, result)
  return value_impl.Value(result)


def ensure_federated_value(
    value: value_impl.Value,
    placement: Optional[placements.PlacementLiteral] = None,
    label: Optional[str] = None,
):
  """Ensures `value` is a federated value placed at `placement`.

  If `value` is not a `computation_types.FederatedType` but is a
  `computation_types.StructType` that can be converted via `federated_zip`
  to a `computation_types.FederatedType`, inserts the call to `federated_zip`
  and returns the result. If `value` cannot be converted, raises a TypeError.

  Args:
    value: A `value_impl.Value` to check and convert to a federated value if
      possible.
    placement: The expected placement. If None, any placement is allowed.
    label: An optional string label that describes `value`.

  Returns:
    The value as a federated value, automatically zipping if necessary.

  Raises:
    TypeError: if `value` is not a `FederatedType` and cannot be converted to
      a `FederatedType` with `federated_zip`.
  """

  if not isinstance(value.type_signature, computation_types.FederatedType):
    comp = value.comp
    try:
      zipped = building_block_factory.create_federated_zip(comp)
    except (TypeError, ValueError) as e:
      raise TypeError(
          'The {l} must be a FederatedType or implicitly convertible '
          'to a FederatedType (got a {t}).'.format(
              l=label if label else 'value', t=comp.type_signature
          )
      ) from e
    value = value_impl.Value(zipped)

  if placement is not None and value.type_signature.placement is not placement:  # pytype: disable=attribute-error
    raise TypeError(
        'The {} should be placed at {}, but it is placed at {}.'.format(
            label if label else 'value',
            placement,
            value.type_signature.placement,  # pytype: disable=attribute-error
        )
    )

  return value
