# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines classes/functions to manipulate the API context stack."""

from collections.abc import Iterator
import contextlib
import threading
from typing import Union

from federated_language.context_stack import context
from federated_language.context_stack import context_stack_base
from federated_language.context_stack import runtime_error_context


_Context = Union[context.AsyncContext, context.SyncContext]


class ContextStackImpl(context_stack_base.ContextStack, threading.local):
  """An implementation of a common thread-local context stack to run against."""

  def __init__(self, default_context: _Context):
    super().__init__()
    self._stack = [default_context]

  def set_default_context(self, ctx: _Context) -> None:
    """Installs `ctx` as the default context at the bottom of the stack."""
    self._stack[0] = ctx

  @property
  def current(self) -> _Context:
    return self._stack[-1]

  @contextlib.contextmanager
  def install(self, ctx: _Context) -> Iterator[_Context]:
    self._stack.append(ctx)
    try:
      yield ctx
    finally:
      self._stack.pop()


context_stack = ContextStackImpl(runtime_error_context.RuntimeErrorContext())


def get_context_stack() -> context_stack_base.ContextStack:
  """Returns the global context stack."""
  return context_stack


def set_default_context(ctx: _Context) -> None:
  """Installs `ctx` as the default context in the global context stack."""
  context_stack.set_default_context(ctx)


def set_no_default_context() -> None:
  context_stack.set_default_context(runtime_error_context.RuntimeErrorContext())
