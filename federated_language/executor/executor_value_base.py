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
"""A base Python interface for values embedded in executors."""

import abc

from federated_language.types import typed_object


class ExecutorValue(typed_object.TypedObject, abc.ABC):
  """Represents the abstract interface for values embedded within executors.

  The embedded values may represent computations in-flight that may materialize
  in the future or fail before they materialize.
  """

  @property
  @abc.abstractmethod
  def reference(self):
    """Returns a reference to the value without transferring ownership.

    A reference is an opaque object that is understood by the executors that
    produced the value, therefore:

    1. Executors need to preserve this contract in their implementation.
    2. Users of Executors should not need to depend on this value.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def compute(self):
    """A coroutine that asynchronously returns the computed form of the value.

    The computed form of a value can take a number of forms, such as primitive
    types in Python, numpy arrays, or even eager tensors in case this is an
    eager executor, or an executor backed by an eager one.

    Returns:
      The computed form of the value, as defined above.
    """
    raise NotImplementedError
