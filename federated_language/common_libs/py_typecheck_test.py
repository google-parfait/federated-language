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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.common_libs import py_typecheck


class IsNameValuePairTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('tuple_unnamed', (None, 1)),
      ('tuple_named', ('a', 1)),
      ('list_unnamed', [None, 1]),
      ('list_named', ['a', 1]),
  ])
  def test_returns_true_with_obj(self, obj):
    actual_result = py_typecheck.is_name_value_pair(obj)
    self.assertTrue(actual_result)

  @parameterized.named_parameters([
      ('sequence_short', ('a',)),
      ('sequence_long', ('a', 'b', 'c')),
      ('sequence_wrong_name_type', (1, 2)),
      ('int', 1),
      ('str', 'a'),
      ('dict', {'a': 1}),
  ])
  def test_returns_false_with_obj(self, obj):
    actual_result = py_typecheck.is_name_value_pair(obj)
    self.assertFalse(actual_result)

  @parameterized.named_parameters([
      ('optional_str_unnamed', (None, 1), Optional[str]),
      ('optional_str_named', ('a', 1), Optional[str]),
      ('str', ('a', 1), str),
      ('none', (None, 1), type(None)),
  ])
  def test_returns_true_with_name_type(self, obj, name_type):
    actual_result = py_typecheck.is_name_value_pair(obj, name_type=name_type)
    self.assertTrue(actual_result)

  @parameterized.named_parameters([
      ('wrong_name_type_str', (None, 1), str),
      ('wrong_name_type_none', ('a', 1), type(None)),
  ])
  def test_returns_false_with_name_type(self, obj, name_type):
    actual_result = py_typecheck.is_name_value_pair(obj, name_type=name_type)
    self.assertFalse(actual_result)

  @parameterized.named_parameters([
      ('int', (None, 1), int),
  ])
  def test_returns_true_with_value_type(self, obj, value_type):
    actual_result = py_typecheck.is_name_value_pair(obj, value_type=value_type)
    self.assertTrue(actual_result)

  @parameterized.named_parameters([
      ('wrong_value_type_str', (None, 1), str),
  ])
  def test_returns_false_with_value_type(self, obj, value_type):
    actual_result = py_typecheck.is_name_value_pair(obj, value_type=value_type)
    self.assertFalse(actual_result)


if __name__ == '__main__':
  absltest.main()
