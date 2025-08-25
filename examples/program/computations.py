# Copyright 2022 Google LLC

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
"""An example of computations to use in a federated program."""

import collections
from collections.abc import Sequence

import federated_language
import numpy as np

from computation import python_computation  # pylint: disable=g-bad-import-order


METRICS_TOTAL_SUM = 'total_sum'


@federated_language.federated_computation()
def initialize():
  """Returns the initial state."""
  return federated_language.federated_value(0, federated_language.SERVER)


@python_computation.python_computation(
    [federated_language.SequenceType(np.int32)],
    np.int32,
)
def _sum_sequence(values: Sequence[int]) -> int:
  """Returns the sum of all the integers in `values`."""
  return sum(values)


@python_computation.python_computation([np.int32, np.int32], np.int32)
def _sum_integers(x: int, y: int) -> int:
  """Returns the sum of two integers."""
  return x + y


@federated_language.federated_computation(
    federated_language.FederatedType(np.int32, federated_language.SERVER),
    federated_language.FederatedType(
        federated_language.SequenceType(np.int32),
        federated_language.CLIENTS,
    ),
)
def train(server_state: int, client_data: Sequence[Sequence[int]]):
  """Computes the sum of all the integers on the clients.

  Computes the sum of all the integers on the clients, updates the server state,
  and returns the updated server state and the following metrics:

  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.

  Args:
    server_state: The server state.
    client_data: The data on the clients.

  Returns:
    A tuple of the updated server state and the train metrics.
  """
  client_sums = federated_language.federated_map(_sum_sequence, client_data)
  total_sum = federated_language.federated_sum(client_sums)
  updated_state = federated_language.federated_map(
      _sum_integers, (server_state, total_sum)
  )
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])
  metrics = federated_language.federated_zip(metrics)
  return updated_state, metrics


@federated_language.federated_computation(
    federated_language.FederatedType(np.int32, federated_language.SERVER),
    federated_language.FederatedType(
        federated_language.SequenceType(np.int32),
        federated_language.CLIENTS,
    ),
)
def evaluation(server_state: int, client_data: Sequence[int]):
  """Computes the sum of all the integers on the clients.

  Computes the sum of all the integers on the clients and returns the following
  metrics:

  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.

  Args:
    server_state: The server state.
    client_data: The data on the clients.

  Returns:
    The evaluation metrics.
  """
  del server_state  # Unused.
  client_sums = federated_language.federated_map(_sum_sequence, client_data)
  total_sum = federated_language.federated_sum(client_sums)
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])
  metrics = federated_language.federated_zip(metrics)
  return metrics
