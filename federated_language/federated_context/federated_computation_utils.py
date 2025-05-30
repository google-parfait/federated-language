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
# limitations under the License.
"""Helpers for creating larger structures out of computing building blocks."""

from typing import Optional

from federated_language.compiler import building_blocks
from federated_language.computation import computation_wrapper
from federated_language.context_stack import context_stack_impl
from federated_language.federated_context import federated_computation_context
from federated_language.federated_context import value_impl
from federated_language.types import computation_types
from federated_language.types import type_conversions


def zero_or_one_arg_fn_to_building_block(
    fn,
    parameter_name: Optional[str],
    parameter_type: Optional[computation_types.Type],
    context_stack: context_stack_impl.ContextStack,
    suggested_name: Optional[str] = None,
) -> tuple[
    building_blocks.ComputationBuildingBlock, computation_types.FunctionType
]:
  """Converts a zero- or one-argument `fn` into a computation building block.

  Args:
    fn: A function with 0 or 1 arguments that contains orchestration logic,
      i.e., that expects zero or one `values_base.Value` and returns a result
      convertible to the same.
    parameter_name: The name of the parameter, or `None` if there is't any.
    parameter_type: The `federated_language.Type` of the parameter, or `None` if
      there's none.
    context_stack: The context stack to use.
    suggested_name: The optional suggested name to use for the federated context
      that will be used to serialize this function's body (ideally the name of
      the underlying Python function). It might be modified to avoid conflicts.

  Returns:
    A tuple of `(building_blocks.ComputationBuildingBlock,
    computation_types.Type)`, where the first element contains the logic from
    `fn`, and the second element contains potentially annotated type information
    for the result of `fn`.

  Raises:
    ValueError: if `fn` is incompatible with `parameter_type`.
  """
  if isinstance(
      context_stack.current,
      federated_computation_context.FederatedComputationContext,
  ):
    parent_context = context_stack.current
  else:
    parent_context = None
  context = federated_computation_context.FederatedComputationContext(
      context_stack, suggested_name=suggested_name, parent=parent_context
  )
  if parameter_name is not None:
    parameter_name = '{}_{}'.format(context.name, str(parameter_name))
  with context_stack.install(context):
    if parameter_type is not None:
      result = fn(
          value_impl.Value(
              building_blocks.Reference(parameter_name, parameter_type),
          )
      )
    else:
      result = fn()
    if result is None:
      raise computation_wrapper.ComputationReturnedNoneError(fn)
    annotated_result_type = type_conversions.infer_type(result)
    result = value_impl.to_value(result, type_spec=annotated_result_type)
    result_comp = result.comp
    symbols_bound_in_context = context_stack.current.symbol_bindings
    if symbols_bound_in_context:
      result_comp = building_blocks.Block(
          local_symbols=symbols_bound_in_context, result=result_comp
      )
    annotated_type = computation_types.FunctionType(
        parameter_type, annotated_result_type
    )
    return (
        building_blocks.Lambda(parameter_name, parameter_type, result_comp),
        annotated_type,
    )
