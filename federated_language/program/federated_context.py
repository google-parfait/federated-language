# Copyright 2022 Google LLC
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
"""Defines an abstract interface for representing a federated context."""

import abc
from typing import Optional, Union

from federated_language.computation import computation_base
from federated_language.context_stack import context
from federated_language.context_stack import context_stack_impl
from federated_language.program import structure_utils
from federated_language.program import value_reference
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_analysis


ComputationArg = Union[
    value_reference.MaterializableStructure,
    object,
    computation_base.Computation,
]


def contains_only_server_placed_data(
    type_signature: computation_types.Type,
) -> bool:
  """Determines if `type_signature` contains only server-placed data.

  Determines if `type_signature` contains only:
  * `federated_language.StructType`s
  * `federated_language.SequenceType`s
  * server-placed `federated_language.FederatedType`s
  * `federated_language.TensorType`s

  Args:
      type_signature: The `federated_language.Type` to test.

  Returns:
    `True` if `type_signature` contains only server-placed data, otherwise
    `False`.
  """

  def _predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.StructType,
            computation_types.SequenceType,
            computation_types.TensorType,
        ),
    ) or (
        isinstance(type_spec, computation_types.FederatedType)
        and type_spec.placement is placements.SERVER
    )

  return type_analysis.contains_only(type_signature, _predicate)


class FederatedContext(context.SyncContext):
  """An abstract interface representing a federated context.

  A federated context supports invoking a limited set of
  `federated_language.Computation`s,
  making guarantees about what a `federated_language.Computation` can accept as
  an argument and
  what it returns when invoked.

  ## Restrictions on the Federated Language Type

  Arguments can be nested structures of values corresponding to the TensorFlow
  Federated type signature of the `federated_language.Computation`:

  * Server-placed values must be represented by
    `federated_language.program.MaterializableStructure`.
  * Client-placed values must be represented by structures of values returned by
    a `federated_language.program.FederatedDataSourceIterator`.

  Return values can be structures of
  `federated_language.program.MaterializableValueReference`s
  or a single `federated_language.program.MaterializableValueReference`, where a
  reference
  corresponds to the tensor-type of the Federated Language type signature in
  the return value of the invoked `federated_language.Computation`.

  ## Federated Language Type to Python Representation

  In order to interact with the value returned by a
  `federated_language.Computation`, it is
  helpful to be able to reason about the Python type of this value. In some way
  this Python type must depend on the Federated Language type signature of the
  associated value. To provide uniformity of experience and ease of reasoning,
  we specify the Python representation of values in a manner that can be stated
  entirely in the Federated Language typesystem.

  We have chosen to limit the Federated Language type signatures of invoked
  `federated_language.Computation`s to disallow the returning of client-placed
  values,
  `federated_language.SequenceTypes`, and `federated_language.FunctionTypes`, in
  order to reduced the area
  which needs to be supported by federated programs. Below we describe the
  mapping between Federated Language type signatures and Python
  representations of values that can be passed as arguments to or returned as
  results from `federated_language.Computation`s.

  Python representations of values that can be *accepted as an arguments to* or
  *returned as a value from* a `federated_language.Computation`:

  | Federated Language Type  | Python Representation                      |
  | ------------------------ | ------------------------------------------ |
  | `TensorType`             | `MaterializableValueReference`             |
  | `SequenceType`           | `MaterializableValueReference`             |
  | `FederatedType`          | Python representation of the `member` of   |
  : (server-placed)          : the `federated_language.FederatedType`     :
  | `StructWithPythonType`   | Python container of the                    |
  :                          : `federated_language.StructWithPythonType`  :
  | `StructType` (with no    | `collections.OrderedDict`                  |
  : Python type, all fields  :                                            :
  : named)                   :                                            :
  | `StructType` (with no    | `tuple`                                    |
  : Python type, no fields   :                                            :
  : named)                   :                                            :

  Python representations of values that can be only be *accepted as an arguments
  to* a `federated_language.Computation`:

  | Federated Language Type  | Python Representation                   |
  | ------------------------ | --------------------------------------- |
  | `FederatedType`          | Opaque object returned by               |
  : (client-placed)          : `DataSourceIterator.select`             :
  | `FunctionType`           | `federated_language.Computation`        |
  """

  @abc.abstractmethod
  def invoke(
      self,
      comp: computation_base.Computation,
      arg: Optional[ComputationArg],
  ) -> structure_utils.Structure[value_reference.MaterializableValueReference]:
    """Invokes the `comp` with the argument `arg`.

    Args:
      comp: The `federated_language.Computation` being invoked.
      arg: The optional argument of `comp`; server-placed values must be
        represented by `federated_language.program.MaterializableStructure`, and
        client-placed values must be represented by structures of values
        returned by a `federated_language.program.FederatedDataSourceIterator`.

    Returns:
      The result of invocation; a structure of
      `federated_language.program.MaterializableValueReference`.

    Raises:
      ValueError: If the result type of `comp` does not contain only structures,
      server-placed values, or tensors.
    """
    raise NotImplementedError


def check_in_federated_context() -> None:
  """Checks if the current context is a `federated_language.program.FederatedContext`."""
  context_stack = context_stack_impl.get_context_stack()
  if not isinstance(context_stack.current, FederatedContext):
    raise ValueError(
        'Expected the current context to be a'
        ' `federated_language.program.FederatedContext`, found'
        f' {type(context_stack.current)}.'
    )
