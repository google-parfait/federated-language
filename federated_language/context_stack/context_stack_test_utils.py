# Copyright 2019 Google LLC
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
"""Utilities for testing context stacks."""

import asyncio
from collections.abc import Callable
import contextlib
import functools
from typing import Union

from federated_language.context_stack import context_base
from federated_language.context_stack import context_stack_impl

_Context = Union[context_base.AsyncContext, context_base.SyncContext]
_ContextFactory = Callable[[], _Context]


class TestContext(context_base.SyncContext):
  """A test context."""

  def invoke(self, comp, arg):
    return NotImplementedError


@contextlib.contextmanager
def test_environment():
  yield None


def with_context(context_fn: _ContextFactory):
  """Returns a decorator for running a test in a context.

  Args:
    context_fn: A `Callable` that constructs a
      `federated_language.framework.AsyncContext` or
      `federated_language.framework.SyncContext` to install before invoking the
      decorated function.
  """

  def decorator(fn):

    @contextlib.contextmanager
    def _install_context(context_fn: _ContextFactory):
      context = context_fn()
      with context_stack_impl.context_stack.install(context):
        yield

    if asyncio.iscoroutinefunction(fn):

      @functools.wraps(fn)
      async def wrapper(*args, **kwargs):
        with _install_context(context_fn):
          return await fn(*args, **kwargs)

    else:

      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        with _install_context(context_fn):
          return fn(*args, **kwargs)

    return wrapper

  return decorator
