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
"""A base Python interface for all types of executors."""

import abc
from typing import Optional

from federated_language.executor import executor_value_base


class RetryableError(Exception):
  """Raised when execution fails and can be retried."""


class Executor(abc.ABC):
  """Represents the abstract interface that all executors must implement."""

  @abc.abstractmethod
  def close(self):
    """Release resources associated with this Executor, if any.

    If the executor has one or more target Executors, implementation of this
    method must close them.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_value(
      self, value, type_spec=None
  ) -> executor_value_base.ExecutorValue:
    """A coroutine that creates embedded value from `value` of type `type_spec`.

    This function is used to embed a value within the executor. The argument
    can be one of the plain Python types, a nested structure, a representation
    of a computation, etc. Once embedded, the value can be further passed
    around within the executor. For functional values, embedding them prior to
    invocation potentially allows the executor to amortize overhead across
    multiple calls.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: An optional `federated_language.Type` of the value represented
        by this object, or something convertible to it. The type can only be
        omitted if the value is a instance of `federated_language.TypedObject`.

    Returns:
      An instance of `ExecutorValue` that represents the embedded value.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_call(
      self,
      comp: executor_value_base.ExecutorValue,
      arg: Optional[executor_value_base.ExecutorValue] = None,
  ) -> executor_value_base.ExecutorValue:
    """A coroutine that creates a call to `comp` with optional argument `arg`.

    Args:
      comp: The computation to invoke. It must have been first embedded in the
        executor by calling `create_value()` on it first.
      arg: An optional argument of the call, or `None` if no argument was
        supplied. If it is present, it must have been embedded in the executor
        by calling `create_value()` on it first.

    Returns:
      An instance of `ExecutorValue` that represents the constructed call.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_struct(self, elements) -> executor_value_base.ExecutorValue:
    """A coroutine that creates a tuple of `elements`.

    Args:
      elements: A collection of `ExecutorValue`s to create a tuple from. The
        collection may be of any kind accepted by `structure.from_container`,
        including dictionaries and lists. The `ExecutorValues` in the container
        must have been created by calling `create_value` on this executor.

    Returns:
      An instance of `ExecutorValue` that represents the constructed tuple.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_selection(
      self, source, index
  ) -> executor_value_base.ExecutorValue:
    """A coroutine that creates a selection from `source`.

    Args:
      source: The source to select from. The source must have been embedded in
        this executor by invoking `create_value()` on it first.
      index: An integer index to select.

    Returns:
      An instance of `ExecutorValue` that represents the constructed selection.
    """
    raise NotImplementedError
