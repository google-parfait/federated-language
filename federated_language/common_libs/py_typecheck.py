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
"""Utility functions for checking Python types."""

from collections.abc import Sequence
import sys
import typing
from typing import Optional, Protocol, TypeVar, Union

from typing_extensions import TypeGuard


@typing.runtime_checkable
class SupportsNamedTuple(Protocol):
  """A `typing.Protocol` with two abstract method `_fields` and `_asdict`."""

  @property
  def _fields(self) -> tuple[str, ...]:
    ...

  def _asdict(self) -> dict[str, object]:
    ...


_NT = TypeVar('_NT', bound=Optional[str])
_VT = TypeVar('_VT', bound=object)


def is_name_value_pair(
    obj: object,
    name_type: type[_NT] = Optional[str],
    value_type: type[_VT] = object,
) -> TypeGuard[tuple[_NT, _VT]]:
  """Returns `True` if `obj` is a name value pair, otherwise `False`.

  In `federated_language`, a name value pair (or named field) is a
  `collection.abc.Sequence` of exactly two elements, a `name` (which can be
  `None`) and a `value`.

  Args:
    obj: The object to test.
    name_type: The type of the name.
    value_type: The type of the value.
  """
  if not isinstance(obj, Sequence) or len(obj) != 2:
    return False
  name, value = obj

  # Before Python 3.10, you could not pass a `Union Type` to isinstance, see
  # https://docs.python.org/3/library/functions.html#isinstance.
  if sys.version_info < (3, 10):

    def _unpack_type(x):
      origin = typing.get_origin(x)
      if origin is Union:
        return typing.get_args(name_type)
      elif origin is not None:
        return origin
      else:
        return x

    name_type = _unpack_type(name_type)
    value_type = _unpack_type(value_type)

  return isinstance(name, name_type) and isinstance(value, value_type)
