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
"""A pipeline that reduces computations into an executable form."""

from collections.abc import Callable
import functools
from typing import Generic, TypeVar

from federated_language.computation import computation_base

_Computation = TypeVar('_Computation', bound=computation_base.Computation)


class CompilerPipeline(Generic[_Computation]):
  """An interface for generating executable artifacts.

  The `CompilerPipeline` holds very little logic; caching for the
  artifacts it generates and essentially nothing else. The `CompilerPipeline`
  is initialized with a `compiler_fn`, to which the pipeline itself delegates
  all the actual work of compilation.

  Different backends may accept different executable artifacts; e.g. a
  backend that supports only a map-reduce execution model may accept instances
  of `federated_language.backends.mapreduce.MapReduceForm`. The representation
  of such a backend takes the form of an instance of
  `federated_language.framework.SyncContext` or
  `federated_language.framework.AsyncContext`, which would be initialized with a
  `CompilerPipeline` whose `compilation_fn` accepts a
  `federated_language.Computation` and returns
  `federated_language.backends.mapreduce.MapReduceForm`s.
  """

  def __init__(self, compiler_fn: Callable[[_Computation], object]):
    self._compiler_fn = compiler_fn

  @functools.lru_cache()
  def compile(self, comp: _Computation) -> object:
    """Compiles `comp`."""
    return self._compiler_fn(comp)
