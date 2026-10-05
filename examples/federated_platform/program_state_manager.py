# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example program state manager to use in a federated program.

This example demonstrates how to implement a
`federated_language.program.ProgramStateManager`.

WARNING: This example uses `pickle` to serialize Python. This purpose of this
example is to highlight concepts of the Federated Language, serializing Python
using `pickle` is not recommended for all systems and should thought of as an
implementation detail.
"""

import os
import os.path
import pickle
from typing import Optional, Union

import federated_language
import tree


class ProgramStateManager(
    federated_language.program.ProgramStateManager[
        federated_language.program.ProgramStateStructure
    ]
):
  """A `ProgramStateManager` that saves program state to memory."""

  def __init__(
      self,
      program_state_dir: Union[str, os.PathLike[str]],
      prefix: str = 'program_state_',
  ):
    if not program_state_dir:
      raise ValueError(
          'Expected `program_state_dir` to not be an empty string.'
      )

    if not os.path.exists(program_state_dir):
      os.makedirs(program_state_dir)

    self._program_state_dir = program_state_dir
    self._prefix = prefix

  async def get_versions(self) -> Optional[list[int]]:
    versions = []
    entries = os.listdir(self._program_state_dir)
    for entry in entries:
      if entry.startswith(self._prefix):
        version_str = entry[len(self._prefix) :]
        version = int(version_str)
        versions.append(version)
    return sorted(versions)

  async def load(
      self,
      version: int,
      structure: federated_language.program.ProgramStateStructure,
  ) -> federated_language.program.ProgramStateStructure:
    path = os.path.join(self._program_state_dir, f'{self._prefix}{version}')
    if not os.path.exists(path):
      raise federated_language.program.ProgramStateNotFoundError(version)

    with open(path, 'rb') as file:
      state = pickle.load(file)

    def _deserialize_as(structure, value):
      if isinstance(structure, federated_language.Serializable):
        serializable_cls = type(structure)
        value = serializable_cls.from_bytes(value)
      return value

    # Note: `map_structure_up_to` is used because `structure` may be shallower
    # than `normalized_state`. For example, a `MaterializableValueReference` may
    # reference a sequence of data.
    deserialized_state = tree.map_structure_up_to(
        structure, _deserialize_as, structure, state
    )
    return deserialized_state

  async def save(
      self,
      program_state: federated_language.program.ProgramStateStructure,
      version: int,
  ) -> None:
    path = os.path.join(self._program_state_dir, f'{self._prefix}{version}')
    if os.path.exists(path):
      os.remove(path)

    materialized_state = await federated_language.program.materialize_value(
        program_state
    )

    def _serialize(value):
      if isinstance(value, federated_language.Serializable):
        value = value.to_bytes()
      return value

    serialized_state = tree.map_structure(_serialize, materialized_state)

    with open(path, 'wb') as file:
      pickle.dump(serialized_state, file)
