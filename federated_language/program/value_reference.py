# Copyright 2021 Google LLC
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
"""Defines abstract interfaces for representing references to values.

These abstract interfaces provide the capability to handle values without
requiring them to be materialized as Python objects. Instances of these
abstract interfaces represent values of type `federated_language.TensorType` and
can be placed
on the server, elements of structures that are placed on the server, or
unplaced.
"""

import abc
import asyncio
from collections.abc import Iterable
from typing import Union

from federated_language.program import structure_utils
from federated_language.types import computation_types
from federated_language.types import typed_object
import numpy as np

MaterializableTypeSignature = Union[
    computation_types.TensorType,
    computation_types.SequenceType,
]
_MaterializedArrayValue = Union[
    # Python types
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    # Numpy types
    np.generic,
    np.ndarray,
]
MaterializedValue = Union[
    _MaterializedArrayValue,
    Iterable[_MaterializedArrayValue],
]
MaterializedStructure = structure_utils.Structure[MaterializedValue]
MaterializableValue = Union[
    MaterializedValue,
    'MaterializableValueReference',
]
MaterializableStructure = structure_utils.Structure[MaterializableValue]


class MaterializableValueReference(typed_object.TypedObject, abc.ABC):
  """An abstract interface representing references to server-placed values."""

  @property
  @abc.abstractmethod
  def type_signature(self) -> MaterializableTypeSignature:
    """The `federated_language.Type` of this object."""
    raise NotImplementedError

  @abc.abstractmethod
  async def get_value(self) -> MaterializedValue:
    """Returns the referenced value.

    The Python type of the referenced value depends on the `type_signature`:

    | Federated Language Type  | Python Type                                  |
    | ------------------------ | -------------------------------------------- |
    | `TensorType`             | `bool`, `int`, `float`, `complex`, `str`,    |
    |                          | `bytes`, `np.generic`, or `np.ndarray`       |
    | `SequenceType`           | `Iterable` of any Python type corresponding  |
    |                          |  to a `federated_language.TensorType`        |
    """
    raise NotImplementedError


async def materialize_value(
    value: MaterializableStructure,
) -> MaterializedStructure:
  """Materializes the `federated_language.program.MaterializableValueReference`s in `value`.

  Args:
    value: A `federated_language.program.MaterializableStructure` to
      materialize.

  Returns:
    A `federated_language.program.MaterializedStructure`.
  """

  async def _materialize(value: MaterializableValue) -> MaterializedValue:
    if isinstance(value, MaterializableValueReference):
      return await value.get_value()
    else:
      return value

  flattened_value = structure_utils.flatten(value)
  materialized_value = await asyncio.gather(
      *[_materialize(v) for v in flattened_value]
  )
  return structure_utils.unflatten_as(value, materialized_value)
