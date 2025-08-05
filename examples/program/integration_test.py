# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example of program logic test."""

import functools
import unittest

from absl.testing import absltest
import federated_language

from algorithm import program_logic  # pylint: disable=g-bad-import-order
from executor import execution_context  # pylint: disable=g-bad-import-order
from federated_platform import data_source  # pylint: disable=g-bad-import-order
from federated_platform import program_state_manager  # pylint: disable=g-bad-import-order
from program import computations  # pylint: disable=g-bad-import-order


def _create_native_federated_context():
  context = execution_context.ExecutionContext()
  return federated_language.program.NativeFederatedContext(context)


class IntegrationTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  @federated_language.framework.with_context(_create_native_federated_context)
  async def test_fault_tolerance(self):
    values = [list(range(10))] * 3
    train_data_source = data_source.DataSource(values)
    evaluation_data_source = data_source.DataSource(values)
    num_clients = 3
    program_state_dir = self.create_tempdir()
    program_state_mngr = program_state_manager.ProgramStateManager(
        program_state_dir
    )

    train_federated_model = functools.partial(
        program_logic.train_federated_model,
        initialize=computations.initialize,
        train=computations.train,
        train_data_source=train_data_source,
        evaluation=computations.evaluation,
        evaluation_data_source=evaluation_data_source,
        num_clients=num_clients,
        program_state_manager=program_state_mngr,
    )

    # Train first round.
    await train_federated_model(total_rounds=1)

    actual_versions = await program_state_mngr.get_versions()
    self.assertEqual(actual_versions, [1])

    # Train second round. This simulates a failure after the first round.
    await train_federated_model(total_rounds=2)

    actual_versions = await program_state_mngr.get_versions()
    self.assertEqual(actual_versions, [1, 2])


if __name__ == '__main__':
  absltest.main()
