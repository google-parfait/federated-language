# Copyright 2025 Google LLC
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
"""An example data source to use in a federated program.

This example demonstrates how to implement a
`federated_language.program.FederatedDataSource` and
`federated_language.program.FederatedDataSourceIterator`.

WARNING: This example uses `pickle` to serialize Python. This purpose of this
example is to highlight concepts of the Federated Language, serializing Python
using `pickle` is not recommended for all systems and should thought of as an
implementation detail.
"""

from collections.abc import Sequence
import pickle
import random
from typing import Optional

import federated_language
import numpy as np


def _check_values(values: Sequence[Sequence[federated_language.Array]]):
  """Checks if `values` are well formed.

  Args:
    values: The values to check.

  Raises:
    ValueError: If `values` is empty or the first sequence in `values` is empty.
  """
  if not values:
    raise ValueError('Expected `values` to not be empty.')

  first_sequence, *_ = values
  if not first_sequence:
    raise ValueError('Expected the first sequence to not be empty.')


class DataSourceIterator(
    federated_language.program.FederatedDataSourceIterator
):
  """A `FederatedDataSourceIterator` backed by `federated_language.Array`s.

  A `federated_language.program.FederatedDataSourceIterator` backed by a
  sequence of `federated_language.Array's, one `federated_language.Array' per
  client. It selects datasources uniformly at random, with replacement over
  successive calls of `select()` but without replacement within a single call of
  `select()`.
  """

  def __init__(self, values: Sequence[Sequence[federated_language.Array]]):
    """Returns an initialized `DataSourceIterator`.

    Args:
      values: A sequence of `federated_language.Array's to use to yield the data
        from this data source.

    Raises:
      ValueError: If `values` is empty.
    """
    _check_values(values)

    first_sequence, *_ = values
    first_value, *_ = first_sequence
    if isinstance(first_value, (np.generic, np.ndarray)):
      dtype = first_value.dtype.type
    else:
      dtype = federated_language.infer_dtype(first_value)

    self._values = values
    self._federated_type = federated_language.FederatedType(
        federated_language.SequenceType(dtype),
        federated_language.CLIENTS,
    )

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'DataSourceIterator':
    """Deserializes the object from bytes."""
    values = pickle.loads(buffer)
    return DataSourceIterator(values)

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    values_bytes = pickle.dumps(self._values)
    return values_bytes

  @property
  def federated_type(self) -> federated_language.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._federated_type

  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of data from this iterator.

    Args:
      k: A number of elements to select. Must be a positive integer and less
        than the number of `values`.

    Raises:
      ValueError: If `k` is not a positive integer or if `k` is not less than
        the number of `values`.
    """
    if k is None or k < 0 or k > len(self._values):
      raise ValueError(
          'Expected `k` to be a positive integer and less than the number of '
          f'`values`, found `k` of {k} and number of `values` of '
          f'{len(self._values)}.'
      )

    return random.sample(self._values, k)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, DataSourceIterator):
      return NotImplemented
    return self._values == other._values


class DataSource(federated_language.program.FederatedDataSource):
  """A `federated_language.program.FederatedDataSource` backed by `federated_language.Array`s.

  A `federated_language.program.FederatedDataSource` backed by a sequence of
  `federated_language.Array's, one `federated_language.Array' per client, and
  selects data uniformly random with replacement.
  """

  def __init__(self, values: Sequence[Sequence[federated_language.Array]]):
    """Returns an initialized `DataSource`.

    Args:
      values: A sequence of `federated_language.Array's to use to yield the data
        from this data source. Must not be empty and each
        `federated_language.Array' must have the same type specification.

    Raises:
      ValueError: If `values` is empty.
    """
    _check_values(values)

    first_sequence, *_ = values
    first_value, *_ = first_sequence
    if isinstance(first_value, (np.generic, np.ndarray)):
      dtype = first_value.dtype.type
    else:
      dtype = federated_language.infer_dtype(first_value)

    self._values = values
    self._federated_type = federated_language.FederatedType(
        federated_language.SequenceType(dtype),
        federated_language.CLIENTS,
    )

  @property
  def federated_type(self) -> federated_language.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  def iterator(self) -> federated_language.program.FederatedDataSourceIterator:
    """Returns a new iterator for retrieving values from this data source."""
    return DataSourceIterator(self._values)
