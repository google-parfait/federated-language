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

from absl.testing import absltest
from federated_language.types import computation_types
from federated_language.types import type_factory
import numpy as np


class TypeConstructorsTest(absltest.TestCase):

  def test_reduction_op(self):
    result_type = computation_types.TensorType(np.float32)
    element_type = computation_types.TensorType(np.int32)
    actual_type = type_factory.reduction_op(result_type, element_type)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([result_type, element_type]), result_type
    )
    self.assertEqual(actual_type, expected_type)

  def test_unary_op(self):
    type_spec = computation_types.TensorType(np.bool_)
    actual_type = type_factory.unary_op(type_spec)
    expected_type = computation_types.FunctionType(np.bool_, np.bool_)
    self.assertEqual(actual_type, expected_type)

  def test_binary_op(self):
    type_spec = computation_types.TensorType(np.bool_)
    actual_type = type_factory.binary_op(type_spec)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([type_spec, type_spec]), type_spec
    )
    self.assertEqual(actual_type, expected_type)


if __name__ == '__main__':
  absltest.main()
