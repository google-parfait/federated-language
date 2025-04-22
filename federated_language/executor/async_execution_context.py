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

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import functools
from typing import Optional

from federated_language.common_libs import retrying
from federated_language.common_libs import structure
from federated_language.common_libs import tracing
from federated_language.computation import computation_base
from federated_language.computation import function_utils
from federated_language.context_stack import context
from federated_language.executor import cardinalities_utils
from federated_language.executor import executor_base
from federated_language.executor import executor_factory  # pylint: disable=unused-import
from federated_language.executor import executor_value_base
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_conversions
import tree


def _is_retryable_error(exception: Exception) -> bool:
  return isinstance(exception, executor_base.RetryableError)


async def _ingest(
    executor: executor_base.Executor,
    val,
    type_spec: computation_types.Type,
) -> executor_value_base.ExecutorValue:
  """A coroutine that handles ingestion.

  Args:
    executor: An instance of `executor_base.Executor`.
    val: The first argument to `AsyncExecutionContext.ingest()`.
    type_spec: The second argument to `AsyncExecutionContext.ingest()`.

  Returns:
    The result of the ingestion.

  Raises:
    TypeError: If the `val` is not a value of type `type_spec`.
  """
  if isinstance(val, executor_value_base.ExecutorValue):
    return val
  elif isinstance(val, structure.Struct) and not isinstance(
      type_spec, computation_types.FederatedType
  ):
    if not isinstance(type_spec, computation_types.StructType):
      raise ValueError(
          f'Expected a `federated_language.StructType`, found {type_spec}.'
      )
    v_elem = structure.to_elements(val)
    t_elem = list(type_spec.items())
    if len(v_elem) != len(t_elem):
      raise TypeError(
          'Value {} does not match type {}: mismatching tuple length.'.format(
              val, type_spec
          )
      )
    for (vk, _), (tk, _) in zip(v_elem, t_elem):
      if vk not in [tk, None]:
        raise TypeError(
            'Value {} does not match type {}: mismatching tuple element '
            'names {} vs. {}.'.format(val, type_spec, vk, tk)
        )
    ingested = []
    for (_, v), (_, t) in zip(v_elem, t_elem):
      ingested.append(_ingest(executor, v, t))
    ingested = await asyncio.gather(*ingested)
    return await executor.create_struct(
        structure.Struct(
            (name, val) for (name, _), val in zip(t_elem, ingested)
        )
    )
  else:
    return await executor.create_value(val, type_spec)


async def _invoke(
    executor: executor_base.Executor,
    comp,
    arg: Optional[executor_value_base.ExecutorValue],
    result_type: computation_types.Type,
) -> object:
  """A coroutine that handles invocation.

  Args:
    executor: An instance of `executor_base.Executor`.
    comp: The first argument to `AsyncExecutionContext.invoke()`.
    arg: The optional second argument to `AsyncExecutionContext.invoke()`.
    result_type: The type signature of the result. This is used to convert the
      execution result into the proper container types.

  Returns:
    The result of the invocation.
  """
  comp = await executor.create_value(
      comp,
      comp.type_signature,  # pytype: disable=attribute-error
  )
  result = await executor.create_call(comp, arg)
  result_value = await result.compute()
  return type_conversions.type_to_py_container(result_value, result_type)


class AsyncExecutionContext(context.AsyncContext):
  """An asynchronous execution context backed by an `executor_base.Executor`.

  This context's `ingest` and `invoke` methods return Python coroutine objects
  which represent the actual work of ingestion and invocation in the backing
  executor.

  This context will support concurrent invocation of multiple computations if
  their arguments have the same cardinalities.
  """

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
    """Initializes an execution context.

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
    super().__init__()
    self._executor_factory = executor_fn
    if compiler_fn is not None:
      cache_decorator = functools.lru_cache()
      self._compiler = cache_decorator(compiler_fn)
    else:
      self._compiler = None
    self._transform_args = transform_args
    self._transform_result = transform_result
    self._cardinality_inference_fn = cardinality_inference_fn

  @property
  def executor_factory(self) -> executor_factory.ExecutorFactory:
    return self._executor_factory

  @contextlib.contextmanager
  def _reset_factory_on_error(
      self,
      cardinalities: Mapping[placements.PlacementLiteral, int],
  ) -> Iterator[executor_base.Executor]:
    try:
      # We pass a copy down to prevent the caller from mutating.
      yield self._executor_factory.create_executor({**cardinalities})
    except Exception:
      self._executor_factory.clean_up_executor({**cardinalities})
      raise

  @retrying.retry(
      retry_on_exception_filter=_is_retryable_error,
      wait_max_ms=30 * 1000,
      wait_multiplier=2,
  )
  async def invoke(self, comp, arg):
    if asyncio.iscoroutine(arg):
      # Awaiting if we are passed a coro allows us to install and use the async
      # context in conjunction with ConcreteComputations' implementation of
      # __call__.
      arg = await arg

    if not isinstance(comp.type_signature, computation_types.FunctionType):
      raise ValueError(
          'Expected a `federated_language.FunctionType`, found'
          f' {comp.type_signature}.'
      )

    if arg is not None and self._transform_args is not None:
      # `transform_args` is not intended to handle `structure.Struct`. Normalize
      # to a Python structure to make it simpler to handle; `args` is sometimes
      # a `structure.Struct` and sometimes it is not, other times it is a Python
      # structure that contains a `structure.Struct`.
      def _to_python(obj):
        if isinstance(obj, structure.Struct):
          return structure.to_odict_or_tuple(obj)
        else:
          return None

      if isinstance(arg, structure.Struct):
        args, kwargs = function_utils.unpack_args_from_struct(arg)
        args = tree.traverse(_to_python, args)
        args = self._transform_args(args)
        if not isinstance(args, Sequence):
          raise ValueError(
              f'Expected `args` to be a `Sequence`, found {type(args)}'
          )
        kwargs = tree.traverse(_to_python, kwargs)
        kwargs = self._transform_args(kwargs)
        if not isinstance(kwargs, Mapping):
          raise ValueError(
              f'Expected `kwargs` to be a `Mapping`, found {type(kwargs)}'
          )
        arg = function_utils.pack_args_into_struct(args, kwargs)
      else:
        arg = tree.traverse(_to_python, arg)
        arg = self._transform_args(arg)

    # Save the type signature before compiling. Compilation currently loses
    # container types, so we must remember them here so that they can be
    # restored in the output.
    result_type = comp.type_signature.result
    if self._compiler is not None:
      with tracing.span('ExecutionContext', 'Compile', span=True):
        comp = self._compiler(comp)

    with tracing.span('ExecutionContext', 'Invoke', span=True):
      if arg is not None:
        cardinalities = self._cardinality_inference_fn(
            arg,
            comp.type_signature.parameter,  # pytype: disable=attribute-error
        )
      else:
        cardinalities = {}

      with self._reset_factory_on_error(cardinalities) as executor:

        if arg is not None:
          arg = await tracing.wrap_coroutine_in_current_trace_context(
              _ingest(
                  executor,
                  arg,
                  comp.type_signature.parameter,  # pytype: disable=attribute-error
              )
          )

        result = await tracing.wrap_coroutine_in_current_trace_context(
            _invoke(executor, comp, arg, result_type)
        )

        if self._transform_result is not None:
          result = self._transform_result(result)
        return result
