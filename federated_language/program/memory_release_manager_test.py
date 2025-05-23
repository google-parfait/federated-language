# Copyright 2021 Google LLC
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

import collections
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.program import memory_release_manager
from federated_language.program import program_test_utils
import numpy as np
import tree


class MemoryReleaseManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'abc', 'abc'),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_array', np.array([1] * 3, np.int32), np.array([1] * 3, np.int32)),
      # materializable value references
      (
          'value_reference_tensor',
          program_test_utils.TestMaterializableValueReference(1),
          1,
      ),
      (
          'value_reference_sequence',
          program_test_utils.TestMaterializableValueReference([1, 2, 3]),
          [1, 2, 3],
      ),
      # serializable values
      (
          'serializable_value',
          program_test_utils.TestSerializable(1, 2),
          program_test_utils.TestSerializable(1, 2),
      ),
      # other values
      (
          'attrs',
          program_test_utils.TestAttrs(1, 2),
          program_test_utils.TestAttrs(1, 2),
      ),
      # structures
      (
          'list',
          [
              True,
              1,
              'abc',
              program_test_utils.TestMaterializableValueReference(2),
              program_test_utils.TestSerializable(3, 4),
          ],
          [
              True,
              1,
              'abc',
              2,
              program_test_utils.TestSerializable(3, 4),
          ],
      ),
      ('list_empty', [], []),
      (
          'list_nested',
          [
              [
                  True,
                  1,
                  'abc',
                  program_test_utils.TestMaterializableValueReference(2),
                  program_test_utils.TestSerializable(3, 4),
              ],
              [5],
          ],
          [
              [
                  True,
                  1,
                  'abc',
                  2,
                  program_test_utils.TestSerializable(3, 4),
              ],
              [5],
          ],
      ),
      (
          'dict_ordered',
          {
              'a': True,
              'b': 1,
              'c': 'abc',
              'd': program_test_utils.TestMaterializableValueReference(2),
              'e': program_test_utils.TestSerializable(3, 4),
          },
          {
              'a': True,
              'b': 1,
              'c': 'abc',
              'd': 2,
              'e': program_test_utils.TestSerializable(3, 4),
          },
      ),
      (
          'dict_unordered',
          {
              'c': True,
              'b': 1,
              'a': 'abc',
              'd': program_test_utils.TestMaterializableValueReference(2),
              'e': program_test_utils.TestSerializable(3, 4),
          },
          {
              'c': True,
              'b': 1,
              'a': 'abc',
              'd': 2,
              'e': program_test_utils.TestSerializable(3, 4),
          },
      ),
      ('dict_empty', {}, {}),
      (
          'dict_nested',
          {
              'x': {
                  'a': True,
                  'b': 1,
                  'c': 'abc',
                  'd': program_test_utils.TestMaterializableValueReference(2),
                  'e': program_test_utils.TestSerializable(3, 4),
              },
              'y': {'a': 5},
          },
          {
              'x': {
                  'a': True,
                  'b': 1,
                  'c': 'abc',
                  'd': 2,
                  'e': program_test_utils.TestSerializable(3, 4),
              },
              'y': {'a': 5},
          },
      ),
      (
          'named_tuple',
          program_test_utils.TestNamedTuple1(
              a=True,
              b=1,
              c='abc',
              d=program_test_utils.TestMaterializableValueReference(2),
              e=program_test_utils.TestSerializable(3, 4),
          ),
          program_test_utils.TestNamedTuple1(
              a=True,
              b=1,
              c='abc',
              d=2,
              e=program_test_utils.TestSerializable(3, 4),
          ),
      ),
      (
          'named_tuple_nested',
          program_test_utils.TestNamedTuple3(
              x=program_test_utils.TestNamedTuple1(
                  a=True,
                  b=1,
                  c='abc',
                  d=program_test_utils.TestMaterializableValueReference(2),
                  e=program_test_utils.TestSerializable(3, 4),
              ),
              y=program_test_utils.TestNamedTuple2(a=5),
          ),
          program_test_utils.TestNamedTuple3(
              x=program_test_utils.TestNamedTuple1(
                  a=True,
                  b=1,
                  c='abc',
                  d=2,
                  e=program_test_utils.TestSerializable(3, 4),
              ),
              y=program_test_utils.TestNamedTuple2(a=5),
          ),
      ),
  )
  async def test_release_saves_value(self, value, expected_value):
    release_mngr = memory_release_manager.MemoryReleaseManager()
    key = 1

    await release_mngr.release(value, key=key)

    self.assertLen(release_mngr._values, 1)
    actual_value = release_mngr._values[1]
    tree.assert_same_structure(actual_value, expected_value)
    program_test_utils.assert_same_key_order(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  async def test_remove_all_with_no_values(self):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    release_mngr.remove_all()

    self.assertEqual(release_mngr._values, {})

  async def test_remove_all_with_values(self):
    release_mngr = memory_release_manager.MemoryReleaseManager()
    release_mngr._values = collections.OrderedDict([(i, i) for i in range(3)])

    release_mngr.remove_all()

    self.assertEqual(release_mngr._values, {})

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('10', 10),
  )
  def test_values_returns_values(self, count):
    expected_values = collections.OrderedDict([(i, i) for i in range(count)])
    release_mngr = memory_release_manager.MemoryReleaseManager()
    release_mngr._values = expected_values

    actual_values = release_mngr.values()

    self.assertEqual(actual_values, expected_values)

  def test_values_returns_copy(self):
    release_mngr = memory_release_manager.MemoryReleaseManager()

    values_1 = release_mngr.values()
    values_2 = release_mngr.values()
    self.assertIsNot(values_1, values_2)


if __name__ == '__main__':
  absltest.main()
