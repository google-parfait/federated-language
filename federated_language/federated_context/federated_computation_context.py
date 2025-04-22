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
# See the License for the specific language governing permissions and
# limitations under the License.
"""The implementation of a context to use in building federated computations."""

from typing import Optional

from federated_language.compiler import building_blocks
from federated_language.context_stack import context_stack_impl
from federated_language.context_stack import symbol_binding_context
from federated_language.federated_context import value_impl
from federated_language.types import computation_types
from federated_language.types import type_conversions


class FederatedComputationContext(
    symbol_binding_context.SymbolBindingContext[
        building_blocks.ComputationBuildingBlock,
        building_blocks.Reference,
    ]
):
  """The context for building federated computations.

  This context additionally holds a list of symbols which are bound to
  `building_block.ComputationBuildingBlocks` during construction of
  `federated_language.Values`, and which respect identical semantics to the
  binding of locals
  in `building_blocks.Blocks`.

  Any `federated_language.Value` constructed in this context may add such a
  symbol binding,
  and thereafter refer to the returned reference in place of the bound
  computation. It is then the responsibility of the installer of this context
  to ensure that the symbols bound during the `federated_language.Value`
  construction process
  are appropriately packaged in the result.
  """

  def __init__(
      self,
      context_stack: context_stack_impl.ContextStack,
      suggested_name: Optional[str] = None,
      parent: Optional['FederatedComputationContext'] = None,
  ):
    """Creates this context.

    Args:
      context_stack: The context stack to use.
      suggested_name: The optional suggested name of the context, a string. It
        may be modified to make it different from the names of any of the
        ancestors on the context stack.
      parent: The optional parent context. If not `None`, it must be an instance
        of `FederatedComputationContext`.
    """
    if suggested_name is None or not suggested_name:
      suggested_name = 'FEDERATED'
    ancestor = parent
    ancestor_names = set()
    while ancestor is not None:
      ancestor_names.add(ancestor.name)
      ancestor = ancestor.parent
    name = suggested_name
    name_count = 0
    while name in ancestor_names:
      name_count = name_count + 1
      name = '{}_{}'.format(suggested_name, name_count)
    self._context_stack = context_stack
    self._parent = parent
    self._name = name
    self._symbol_bindings = []
    self._next_symbol_val = 0

  @property
  def name(self):
    return self._name

  @property
  def parent(self):
    return self._parent

  def bind_computation_to_reference(
      self, comp: building_blocks.ComputationBuildingBlock
  ) -> building_blocks.Reference:
    """Binds a computation to a symbol, returns a reference to this binding."""
    name = 'fc_{name}_symbol_{val}'.format(
        name=self._name, val=self._next_symbol_val
    )
    self._next_symbol_val += 1
    self._symbol_bindings.append((name, comp))
    ref = building_blocks.Reference(name, comp.type_signature)
    return ref

  @property
  def symbol_bindings(
      self,
  ) -> list[tuple[str, building_blocks.ComputationBuildingBlock]]:
    return self._symbol_bindings

  def invoke(self, comp, arg):
    fn = value_impl.to_value(comp, type_spec=None)
    type_spec = fn.type_signature
    if not isinstance(type_spec, computation_types.FunctionType):
      raise ValueError(
          f'Expected {type_spec} to be a `federated_language.FunctionType`.'
      )
    if arg is not None:
      if type_spec.parameter is None:
        raise ValueError(f'Expected no arguments, found {arg}.')
      arg = value_impl.to_value(
          arg, type_spec=type_spec.parameter, zip_if_needed=True
      )
      result = fn(arg)
    else:
      if type_spec.parameter is not None:
        raise ValueError(
            f'Expected an argument of type {type_spec.parameter}, found none.'
        )
      result = fn()

    value_type = type_conversions.infer_type(result)
    if not type_spec.result.is_assignable_from(value_type):
      raise computation_types.TypeNotAssignableError(value_type, type_spec)

    return result
