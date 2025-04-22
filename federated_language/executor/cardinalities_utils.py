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
"""Utilities for cardinality inference and handling."""

from collections.abc import Callable, Mapping, Sequence

from federated_language.types import computation_types
from federated_language.types import placements


def _merge_cardinalities(
    existing: Mapping[placements.PlacementLiteral, int],
    update: Mapping[placements.PlacementLiteral, int],
) -> Mapping[placements.PlacementLiteral, int]:
  """Returns the merged cardinalities after checking for conflicts."""
  if not update:
    return existing
  elif not existing:
    return update

  for k, v in update.items():
    if k in existing and existing[k] != v:
      raise ValueError(
          f'Conflicting cardinalities for {k}: {v} vs {existing[k]}'
      )

  cardinalities = {}
  cardinalities.update(existing)
  cardinalities.update(update)
  return cardinalities


class InvalidNonAllEqualValueError(TypeError):

  def __init__(self, value, type_spec):
    message = (
        f'Expected non-all-equal value with placement {type_spec.placement} '
        'to be a `list` or `tuple`, found a value of Python type '
        f'{type(value)}:\n{value}'
    )
    super().__init__(message)


# We define this type here to avoid having to redeclare it wherever we
# parameterize by a cardinality inference fn.
CardinalityInferenceFnType = Callable[
    [object, computation_types.Type],
    Mapping[placements.PlacementLiteral, int],
]


def infer_cardinalities(
    value: object, type_spec: computation_types.Type
) -> dict[placements.PlacementLiteral, int]:
  """Infers cardinalities from Python `value`.

  Allows for any Python object to represent a federated value; enforcing
  particular representations is not the job of this inference function, but
  rather ingestion functions lower in the stack.

  Args:
    value: Python object from which to infer placement cardinalities.
    type_spec: The type spec for `value`, determining the semantics for
      inferring cardinalities. That is, we only pull the cardinality off of
      federated types.

  Returns:
    Dict of cardinalities.

  Raises:
    ValueError: If conflicting cardinalities are inferred from `value`.
    TypeError: If the arguments are of the wrong types, or if `type_spec` is
      a federated type which is not `all_equal` but the yet-to-be-embedded
      `value` is not represented as a Python `list`.
  """
  if value is None:
    return {}

  if isinstance(type_spec, computation_types.FederatedType):
    if type_spec.all_equal:
      return {}
    if not isinstance(value, Sequence):
      raise InvalidNonAllEqualValueError(value, type_spec)
    return {type_spec.placement: len(value)}
  elif isinstance(type_spec, computation_types.StructType):
    cardinalities = {}

    if isinstance(value, Mapping):
      elements = value.values()
    else:
      elements = value

    for element, (_, element_type) in zip(elements, type_spec.items()):
      update = infer_cardinalities(element, element_type)
      cardinalities = _merge_cardinalities(cardinalities, update)
    return cardinalities
  else:
    return {}
