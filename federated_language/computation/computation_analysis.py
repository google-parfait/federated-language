# Copyright 2025 Google LLC
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
# See the License for the specific language governing permisions and
# limitations under the License.
"""A library of static analysis functions for computations."""

from collections.abc import Callable

from federated_language.compiler import building_block_analysis
from federated_language.compiler import building_blocks
from federated_language.compiler import intrinsic_defs
from federated_language.compiler import tree_analysis
from federated_language.computation import computation_impl


def computation_contains(
    computation: computation_impl.ConcreteComputation,
    predicate: Callable[[building_blocks.ComputationBuildingBlock], bool],
) -> bool:
  """Returns whether or not a building block in `tree` matches `predicate`."""
  building_block = computation.to_building_block()
  return tree_analysis.count(building_block, predicate) != 0


def computation_contains_secure_aggregation(
    computation: computation_impl.ConcreteComputation,
):
  """Determins if `computation` contains a secure aggregation call."""
  predicate = lambda x: building_block_analysis.is_called_aggregation(
      x,
      kind=intrinsic_defs.AggregationKind.SECURE,
  )
  return computation_contains(computation, predicate)


def computation_contains_unsecure_aggregation(
    computation: computation_impl.ConcreteComputation,
):
  """Determins if `computation` contains a secure aggregation call."""
  predicate = lambda x: building_block_analysis.is_called_aggregation(
      x,
      kind=intrinsic_defs.AggregationKind.DEFAULT,
  )
  return computation_contains(computation, predicate)
