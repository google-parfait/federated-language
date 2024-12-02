# Copyright 2018 Google LLC
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

from absl.testing import absltest
from federated_language.compiler import computation_factory
from federated_language.computation import computation_base
from federated_language.computation import computation_impl
from federated_language.computation import computation_serialization
from federated_language.context_stack import context_stack_impl
from federated_language.types import computation_types
import numpy as np


class ComputationSerializationTest(absltest.TestCase):

  def test_serialize_deserialize_round_trip(self):
    type_spec = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(type_spec)
    comp = computation_impl.ConcreteComputation(
        computation_proto=proto,
        context_stack=context_stack_impl.context_stack,
    )
    serialized_comp = computation_serialization.serialize_computation(comp)
    deserialize_comp = computation_serialization.deserialize_computation(
        serialized_comp
    )
    self.assertIsInstance(deserialize_comp, computation_base.Computation)
    self.assertEqual(deserialize_comp, comp)


if __name__ == '__main__':
  absltest.main()
