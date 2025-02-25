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
from federated_language.compiler import building_blocks
from federated_language.computation import computation_impl
from federated_language.context_stack import context_stack_impl
from federated_language.proto import computation_pb2
from federated_language.types import computation_types
import numpy as np


class ConcreteComputationTest(absltest.TestCase):

  def test_something(self):

    # At the moment, this should succeed, as both the computation body and the
    # type are well-formed.
    computation_impl.ConcreteComputation(
        computation_proto=computation_pb2.Computation(**{
            'type': computation_types.FunctionType(
                np.int32, np.int32
            ).to_proto(),
            'intrinsic': computation_pb2.Intrinsic(uri='whatever'),
        }),
        context_stack=context_stack_impl.context_stack,
    )

    # This should fail, as the proto is not well-formed.
    with self.assertRaises(NotImplementedError):
      computation_impl.ConcreteComputation(
          computation_proto=computation_pb2.Computation(),
          context_stack=context_stack_impl.context_stack,
      )

    # This should fail, as "10" is not an instance of
    # `computation_pb2.Computation`.
    with self.assertRaises(TypeError):
      computation_impl.ConcreteComputation(
          computation_proto=10,
          context_stack=context_stack_impl.context_stack,
      )


class FromBuildingBlockTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_impl.ConcreteComputation.from_building_block(None)

  def test_converts_building_block_to_computation(self):
    buiding_block = building_blocks.Lambda(
        'x', np.int32, building_blocks.Reference('x', np.int32)
    )
    computation = computation_impl.ConcreteComputation.from_building_block(
        buiding_block
    )
    self.assertIsInstance(computation, computation_impl.ConcreteComputation)


if __name__ == '__main__':
  absltest.main()
