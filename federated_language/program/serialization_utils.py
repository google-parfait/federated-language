# Copyright 2023 Google LLC
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
"""Utilities for packing and unpacking bytes.

This library primarily uses `struct` to pack and unpack bytes. All the format
strings in this library use the network byte order, the assumption is that this
should be safe as long as both the pack and unpack functions use the same byte
order. See
https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment for
more information.

Important: This library only uses `pickle` to serialize Python containers (e.g.
`Sequence`, `Mapping`, `NamedTuple`, etc) and does not use `pickle` to serialize
the values held in those containers.
"""

from collections.abc import Sequence
import importlib
import struct
from typing import Protocol, TypeVar

from federated_language.common_libs import serializable
from federated_language.proto import computation_pb2
from federated_language.types import computation_types


# The maximum size allowed for serialized `tf.data.Dataset`s.
_MAX_SERIALIZED_DATASET_SIZE = 100 * (1024**2)  # 100 MB


_T = TypeVar('_T')


class PackFn(Protocol[_T]):

  def __call__(self, _: _T) -> bytes:
    ...


class UnpackFn(Protocol[_T]):

  def __call__(self, buffer: bytes, offset: int = 0) -> tuple[_T, int]:
    ...


def _pack_length(buffer: bytes) -> bytes:
  """Packs the length of `buffer` as bytes."""
  length = len(buffer)
  length_bytes = struct.pack('!Q', length)
  return length_bytes


def _unpack_length_from(buffer: bytes, offset: int = 0) -> tuple[int, int]:
  """Unpacks a length from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked length and the packed bytes size.
  """
  length_size = struct.calcsize('!Q')
  length, *_ = struct.unpack_from('!Q', buffer, offset=offset)
  return length, length_size


def pack_str(value: str) -> bytes:
  """Packs a `str` as bytes."""
  str_bytes = value.encode('utf-8')
  length_bytes = _pack_length(str_bytes)
  return length_bytes + str_bytes


def unpack_str_from(buffer: bytes, offset: int = 0) -> tuple[str, int]:
  """Unpacks a `str` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `str` and the packed bytes size.
  """
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  str_bytes, *_ = struct.unpack_from(f'!{length}s', buffer, offset=offset)
  value = str_bytes.decode('utf-8')
  return value, length_size + length


def pack_sequence(fn: PackFn[_T], sequence: Sequence[_T]) -> bytes:
  """Packs a `Sequence` as bytes using `fn` to pack each item."""
  sequence_bytes = bytearray()
  for item in sequence:
    item_bytes = fn(item)
    sequence_bytes.extend(item_bytes)
  length_bytes = _pack_length(sequence_bytes)
  return length_bytes + sequence_bytes


def unpack_sequence_from(
    fn: UnpackFn[_T], buffer: bytes, offset: int = 0
) -> tuple[Sequence[_T], int]:
  """Unpacks a `Sequence` from bytes using `fn` to unpack each item.

  Args:
    fn: The `UnpackFn` to use to unpack each item.
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `Sequence` and the packed bytes size.
  """
  sequence = []
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  item_offset = 0
  while item_offset < length:
    item, item_size = fn(buffer, offset=offset + item_offset)
    item_offset += item_size
    sequence.append(item)
  return sequence, length_size + length


def pack_serializable(value: serializable.Serializable) -> bytes:
  """Packs a `federated_language.Serializable` as bytes."""
  module_name_bytes = pack_str(type(value).__module__)
  class_name_bytes = pack_str(type(value).__name__)
  serializable_bytes = value.to_bytes()
  serializable_length_bytes = _pack_length(serializable_bytes)
  return (
      module_name_bytes
      + class_name_bytes
      + serializable_length_bytes
      + serializable_bytes
  )


def unpack_serializable_from(
    buffer: bytes, offset: int = 0
) -> tuple[serializable.Serializable, int]:
  """Unpacks a `federated_language.Serializable` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `federated_language.Serializable` and the
    packed bytes
    size.
  """
  module_name, module_name_size = unpack_str_from(buffer, offset=offset)
  offset += module_name_size
  class_name, class_name_size = unpack_str_from(buffer, offset=offset)
  offset += class_name_size
  serializable_length, serializable_length_size = _unpack_length_from(
      buffer, offset=offset
  )
  offset += serializable_length_size
  serializable_bytes, *_ = struct.unpack_from(
      f'!{serializable_length}s', buffer, offset=offset
  )
  module = importlib.import_module(module_name)
  cls = getattr(module, class_name)
  value = cls.from_bytes(serializable_bytes)
  return value, (
      module_name_size
      + class_name_size
      + serializable_length_size
      + serializable_length
  )


def pack_type_spec(type_spec: computation_types.Type) -> bytes:
  """Packs a `federated_language.Type` as bytes."""
  proto = type_spec.to_proto()
  type_bytes = proto.SerializeToString()
  length_bytes = _pack_length(type_bytes)
  return length_bytes + type_bytes


def unpack_type_spec_from(
    buffer: bytes, offset: int = 0
) -> tuple[computation_types.Type, int]:
  """Unpacks a `federated_language.Type` from bytes.

  Args:
    buffer: The `bytes` to unpack.
    offset: The position in `buffer` to start unpacking from.

  Returns:
    A `tuple` containing the unpacked `federated_language.Type` and the packed
    bytes size.
  """
  length, length_size = _unpack_length_from(buffer, offset=offset)
  offset += length_size
  type_spec_bytes, *_ = struct.unpack_from(f'!{length}s', buffer, offset=offset)
  proto = computation_pb2.Type.FromString(type_spec_bytes)
  type_spec = computation_types.Type.from_proto(proto)
  return type_spec, length_size + length
