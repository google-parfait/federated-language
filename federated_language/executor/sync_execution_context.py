# Copyright 2019 Google LLC
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
"""A context for execution based on an embedded executor instance."""

from collections.abc import Callable
from typing import Optional

from federated_language.common_libs import async_utils
from federated_language.computation import computation_base
from federated_language.context_stack import context
from federated_language.executor import async_execution_context
from federated_language.executor import cardinalities_utils
from federated_language.executor import executor_factory  # pylint: disable=unused-import


class SyncExecutionContext(context.SyncContext):
  """A synchronous execution context backed by an `executor_base.Executor`."""

  def __init__(
      self,
      executor_fn: executor_factory.ExecutorFactory,
      compiler_fn: Optional[
          Callable[[computation_base.Computation], object]
      ] = None,
      *,
      transform_args: Optional[Callable[[object], object]] = None,
      transform_result: Optional[Callable[[object], object]] = None,
      cardinality_inference_fn: cardinalities_utils.CardinalityInferenceFnType = cardinalities_utils.infer_cardinalities,
  ):
    """Initializes a synchronous execution context which retries invocations.

    Args:
      executor_fn: Instance of `executor_factory.ExecutorFactory`.
      compiler_fn: A Python function that will be used to compile a computation.
      transform_args: An `Optional` `Callable` used to transform the args before
        they are passed to the computation.
      transform_result: An `Optional` `Callable` used to transform the result
        before it is returned.
      cardinality_inference_fn: A Python function specifying how to infer
        cardinalities from arguments (and their associated types). The value
        returned by this function will be passed to the `create_executor` method
        of `executor_fn` to construct a `federated_language.framework.Executor`
        instance.
    """
    self._executor_factory = executor_fn
    self._async_context = async_execution_context.AsyncExecutionContext(
        executor_fn=executor_fn,
        compiler_fn=compiler_fn,
        transform_args=transform_args,
        transform_result=transform_result,
        cardinality_inference_fn=cardinality_inference_fn,
    )
    self._async_runner = async_utils.AsyncThreadRunner()

  @property
  def executor_factory(self) -> executor_factory.ExecutorFactory:
    return self._executor_factory

  def invoke(self, comp, arg):
    return self._async_runner.run_coro_and_return_result(
        self._async_context.invoke(comp, arg)
    )
