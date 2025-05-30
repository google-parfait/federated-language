# Copyright 2020 Google LLC
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
"""Defines interface for contexts which can bind symbols."""

import abc
from typing import Generic, TypeVar

from federated_language.context_stack import context


_Symbol = TypeVar('_Symbol')
_Reference = TypeVar('_Reference')


class SymbolBindingContext(
    context.SyncContext, abc.ABC, Generic[_Symbol, _Reference]
):
  """Interface for contexts which handle binding and tracking of references."""

  @abc.abstractmethod
  def bind_computation_to_reference(self, comp: _Symbol) -> _Reference:
    """Binds a computation to a symbol, returns a reference to this binding."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def symbol_bindings(self) -> list[tuple[str, _Symbol]]:
    """Returns all symbols bound in this context."""
    raise NotImplementedError
