# Copyright 2025 Google LLC
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

from typing import NamedTuple
import unittest

from absl.testing import absltest
import federated_language
import numpy as np

import python_computation


class ComputationTest(unittest.TestCase):

  def test_returns_computation_with_parameter_types_none(self):

    @python_computation.python_computation(None, np.int32)
    def _fn():
      return 1

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(None, np.int32)
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_empty(self):

    @python_computation.python_computation([], np.int32)
    def _fn():
      return 1

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(None, np.int32)
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_one_int(self):

    @python_computation.python_computation([np.int32], np.int32)
    def _fn(x):
      return x + 1

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(np.int32, np.int32)
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_one_list(self):

    @python_computation.python_computation(
        [[np.int32, np.float32]],
        [np.int32, np.float32],
    )
    def _fn(x):
      return [x[0] + 1, x[1] + 1.0]

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(
        [np.int32, np.float32],
        [np.int32, np.float32],
    )
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_one_dict(self):

    @python_computation.python_computation(
        [[('a', np.int32), ('b', np.float32)]],
        [('a', np.int32), ('b', np.float32)],
    )
    def _fn(x):
      return {
          'a': x['a'] + 1,
          'b': x['b'] + 1.0,
      }

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(
        [('a', np.int32), ('b', np.float32)],
        [('a', np.int32), ('b', np.float32)],
    )
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_one_named_tuple(
      self,
  ):

    class _Foo(NamedTuple):
      a: int
      b: float

    @python_computation.python_computation([_Foo], _Foo)
    def _fn(x):
      return _Foo(x.a + 1, x.b + 1.0)

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(_Foo, _Foo)
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_multiple_unnamed(self):

    @python_computation.python_computation(
        [np.int32, np.float32],
        [np.int32, np.float32],
    )
    def _fn(x, y):
      return [x + 1, y + 1.0]

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(
        [np.int32, np.float32],
        [np.int32, np.float32],
    )
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_returns_computation_with_parameter_types_multiple_named(self):

    @python_computation.python_computation(
        {'x': np.int32, 'y': np.float32},
        [np.int32, np.float32],
    )
    def _fn(x, y):
      return [x + 1, y + 1.0]

    self.assertIsInstance(_fn, federated_language.framework.ConcreteComputation)
    expected_type = federated_language.FunctionType(
        {'x': np.int32, 'y': np.float32},
        [np.int32, np.float32],
    )
    self.assertEqual(_fn.type_signature, expected_type)
    comp_pb = _fn.to_proto()
    self.assertEqual(comp_pb.type, expected_type.to_proto())
    self.assertTrue(comp_pb.HasField('data'))

  def test_raises_value_error_with_parameter_types_one(self):

    with self.assertRaises(ValueError):

      @python_computation.python_computation([np.int32], np.int32)
      def _fn():
        return 1

  def test_raises_value_error_with_parameter_types_none(self):

    with self.assertRaises(ValueError):

      @python_computation.python_computation(None, np.int32)
      def _fn(x):
        return x + 1

  def test_raises_value_error_with_parameter_types_empty(self):

    with self.assertRaises(ValueError):

      @python_computation.python_computation([], np.int32)
      def _fn(x):
        return x + 1

  def test_wraps_python_function(self):

    @python_computation.python_computation([np.int32], np.int32)
    def _fn(x):
      """Test docstring."""
      return x

    self.assertEqual(_fn.__name__, '_fn')
    self.assertEqual(_fn.__doc__, 'Test docstring.')


if __name__ == '__main__':
  absltest.main()
