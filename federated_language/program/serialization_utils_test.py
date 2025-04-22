# Copyright 2023 Google LLC
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

import struct
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.program import program_test_utils
from federated_language.program import serialization_utils
from federated_language.types import computation_types
import numpy as np


class _TestNamedTuple(NamedTuple):
  a: object
  b: object
  c: object


class SerializationUtilsStrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', ''),
      ('short', 'abc'),
      ('long', 'abc' * 100),
  )
  def test_pack_and_unpack_str(self, value):
    value_bytes = serialization_utils.pack_str(value)
    actual_value, actual_size = serialization_utils.unpack_str_from(value_bytes)

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_str_with_offset(self):
    value = 'abc'
    offset = 100

    value_bytes = serialization_utils.pack_str(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_str, actual_size = serialization_utils.unpack_str_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_str, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_str_from_raises_struct_error_with_offset(self, offset):
    value = 'abc'
    value_bytes = serialization_utils.pack_str(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_str_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_str_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = 'abc'
    value_bytes = serialization_utils.pack_str(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_str_from(corrupt_bytes)


class SerializationUtilsSequenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', []),
      ('short', ['abc', 'def', 'ghi']),
      ('long', ['abc', 'def', 'ghi'] * 100),
  )
  def test_pack_and_unpack_sequence(self, value):
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    actual_value, actual_size = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_str_from, value_bytes
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_sequence_offset(self):
    value = ['abc', 'def', 'ghi']
    offset = 100

    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_str_from, padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_sequence_from_raises_struct_error_with_offset(self, offset):
    value = ['abc', 'def', 'ghi']
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )

    with self.assertRaises(struct.error):
      serialization_utils.unpack_sequence_from(
          serialization_utils.unpack_str_from, value_bytes, offset
      )

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_sequence_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = ['abc', 'def', 'ghi']
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_sequence_from(
          serialization_utils.unpack_str_from, corrupt_bytes
      )


class SerializationUtilsSerializableTest(parameterized.TestCase):

  def test_pack_and_unpack_serializable(self):
    value = program_test_utils.TestSerializable(1, 2)

    value_bytes = serialization_utils.pack_serializable(value)
    actual_value, actual_size = serialization_utils.unpack_serializable_from(
        value_bytes
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_serializable_with_offset(self):
    value = program_test_utils.TestSerializable(1, 2)
    offset = 100

    value_bytes = serialization_utils.pack_serializable(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_serializable_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_serializable_from_raises_struct_error_with_offset(
      self, offset
  ):
    value = program_test_utils.TestSerializable(1, 2)
    value_bytes = serialization_utils.pack_serializable(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_serializable_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_serializable_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = program_test_utils.TestSerializable(1, 2)
    value_bytes = serialization_utils.pack_serializable(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_serializable_from(corrupt_bytes)


class SerializationUtilsTypeSpecTest(parameterized.TestCase):

  def test_pack_and_unpack_type_spec(self):
    value = computation_types.TensorType(np.int32)

    value_bytes = serialization_utils.pack_type_spec(value)
    actual_value, actual_size = serialization_utils.unpack_type_spec_from(
        value_bytes,
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_type_spec_with_offset(self):
    value = computation_types.TensorType(np.int32)
    offset = 100

    value_bytes = serialization_utils.pack_type_spec(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_type_spec_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_type_spec_from_raises_struct_error_with_offset(self, offset):
    value = computation_types.TensorType(np.int32)
    value_bytes = serialization_utils.pack_type_spec(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_type_spec_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_type_spec_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = computation_types.TensorType(np.int32)
    value_bytes = serialization_utils.pack_type_spec(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_type_spec_from(corrupt_bytes)


if __name__ == '__main__':
  absltest.main()
