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
"""Defines the abstract interface for classes that represent computations."""

import abc

from federated_language.types import computation_types
from federated_language.types import typed_object


class Computation(typed_object.TypedObject, abc.ABC):
  """An abstract interface for all classes that represent computations."""

  @property
  @abc.abstractmethod
  def type_signature(self) -> computation_types.FunctionType:
    """Returns the type of this object."""
    raise NotImplementedError

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    """Invokes the computation with the given arguments in the given context.

    Args:
      *args: The positional arguments.
      **kwargs: The keyword-based arguments.

    Returns:
      The result of invoking the computation, the exact form of which depends
      on the context.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hashes the computation.

    Backends reserve the right to compile instances of
    `federated_language.Computation`, as they may need different
    representations or data structures altogether. As these backends need to be
    able to cache the result of compilation, we require that
    `federated_language.Computation` subclasses be hashable.

    Returns:
      Integer representing the hash value of the
      `federated_language.Computation`.
    """
    raise NotImplementedError
