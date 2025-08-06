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
"""Representation of values inside a federated computation."""

import abc
from collections.abc import Hashable, Mapping
import itertools
import typing
from typing import Optional, Union

import attrs
from federated_language.common_libs import py_typecheck
from federated_language.common_libs import structure
from federated_language.compiler import array
from federated_language.compiler import building_block_factory
from federated_language.compiler import building_blocks
from federated_language.computation import computation_impl
from federated_language.computation import function_utils
from federated_language.computation import polymorphic_computation
from federated_language.context_stack import context
from federated_language.context_stack import context_stack_impl
from federated_language.context_stack import symbol_binding_context
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_conversions
from federated_language.types import typed_object


def _unfederated(type_signature):
  if isinstance(type_signature, computation_types.FederatedType):
    return type_signature.member
  return type_signature


def _is_federated_struct(type_spec: computation_types.Type) -> bool:
  return isinstance(type_spec, computation_types.FederatedType) and isinstance(
      type_spec.member, computation_types.StructType
  )


def _check_struct_or_federated_struct(
    vimpl: 'Value',
    attribute: str,
):
  if not isinstance(
      vimpl.type_signature, computation_types.StructType
  ) and not _is_federated_struct(vimpl.type_signature):
    raise AttributeError(
        '`federated_language.Value` of non-structural type'
        f' {vimpl.type_signature} has no attribute {attribute}'
    )


def _bind_computation_to_reference(comp, op: str):
  current_context = context_stack_impl.context_stack.current
  if not isinstance(
      current_context, symbol_binding_context.SymbolBindingContext
  ):
    raise context.ContextError(
        '`federated_language.Value`s should only be used in contexts which can'
        ' bind references, generally a `FederatedComputationContext`.'
        f' Attempted to bind the result of {op} in a context'
        f' {current_context} of type {type(current_context)}.'
    )
  return current_context.bind_computation_to_reference(comp)


class Value(typed_object.TypedObject, abc.ABC):
  """A generic base class for values that appear in computations.

  If the value in this class is of `StructType` or `FederatedType` containing a
  `StructType`, the inner fields can be accessed by name
  (e.g. `y = my_value_impl.y`).
  """

  def __init__(
      self,
      comp: building_blocks.ComputationBuildingBlock,
  ):
    """Constructs a value of the given type.

    Args:
      comp: An instance of building_blocks.ComputationBuildingBlock that
        contains the logic that computes this value.
    """
    super()
    self._comp = comp

  @property
  def type_signature(self):
    return self._comp.type_signature

  @property
  def comp(self) -> building_blocks.ComputationBuildingBlock:
    return self._comp

  def __repr__(self):
    return repr(self._comp)

  def __str__(self):
    return str(self._comp)

  def __dir__(self):
    attributes = ['type_signature', 'comp']
    type_signature = _unfederated(self.type_signature)
    if isinstance(type_signature, computation_types.StructType):
      attributes.extend(dir(type_signature))
    return attributes

  def __getattr__(self, name: str) -> 'Value':
    _check_struct_or_federated_struct(self, name)
    if _is_federated_struct(self.type_signature):
      if name not in self.type_signature.member.fields():  # pytype: disable=attribute-error
        raise AttributeError(
            f"There is no such attribute '{name}' in this federated tuple."
            f" Valid attributes: ({', '.join(dir(self.type_signature.member))})"  # pytype: disable=attribute-error
        )

      return Value(
          building_block_factory.create_federated_getattr_call(self._comp, name)
      )
    if name not in dir(self.type_signature):
      attributes = ', '.join(dir(self.type_signature))
      raise AttributeError(
          f"There is no such attribute '{name}' in this tuple. Valid "
          f'attributes: ({attributes})'
      )
    if isinstance(self._comp, building_blocks.Struct):
      return Value(getattr(self._comp, name))
    return Value(building_blocks.Selection(self._comp, name=name))

  def __bool__(self):
    raise TypeError(
        'Federated computation values do not support boolean operations. '
        'If you were attempting to perform logic on tensors, consider moving '
        'this logic into a `federated_language.tensorflow.computation`.'
    )

  def __len__(self):
    type_signature = _unfederated(self.type_signature)
    if not isinstance(type_signature, computation_types.StructType):
      raise TypeError(
          'Operator len() is only supported for (possibly federated) structure '
          'types, but the object on which it has been invoked is of type {}.'
          .format(self.type_signature)
      )
    return len(type_signature)

  def __getitem__(self, key: Union[int, str, slice]):
    if isinstance(key, str):
      return getattr(self, key)
    if _is_federated_struct(self.type_signature):
      return Value(
          building_block_factory.create_federated_getitem_call(self._comp, key),
      )
    if not isinstance(self.type_signature, computation_types.StructType):
      raise TypeError(
          'Operator getitem() is only supported for structure types, but the '
          'object on which it has been invoked is of type {}.'.format(
              self.type_signature
          )
      )
    elem_length = len(self.type_signature)
    if isinstance(key, int):
      if key < 0 or key >= elem_length:
        raise IndexError(
            'The index of the selected element {} is out of range.'.format(key)
        )
      if isinstance(self._comp, building_blocks.Struct):
        return Value(self._comp[key])
      else:
        return Value(building_blocks.Selection(self._comp, index=key))
    elif isinstance(key, slice):
      index_range = range(*key.indices(elem_length))
      if not index_range:
        raise IndexError(
            'Attempted to slice 0 elements, which is not currently supported.'
        )
      return to_value([self[k] for k in index_range], None)

  def __iter__(self):
    type_signature = _unfederated(self.type_signature)
    if not isinstance(type_signature, computation_types.StructType):
      raise TypeError(
          'Operator iter() is only supported for (possibly federated) '
          'structure types, but the object on which it has been invoked is of '
          f'type {self.type_signature}.'
      )
    for index in range(len(type_signature)):
      yield self[index]

  def __call__(self, *args, **kwargs):
    if not isinstance(self.type_signature, computation_types.FunctionType):
      raise SyntaxError(
          'Function-like invocation is only supported for values of functional '
          'types, but the value being invoked is of type {} that does not '
          'support invocation.'.format(self.type_signature)
      )
    if args or kwargs:
      args = [to_value(x, None) for x in args]
      kwargs = {k: to_value(v, None) for k, v in kwargs.items()}
      arg = function_utils.pack_args(
          self.type_signature.parameter, args, kwargs
      )
      arg = to_value(arg, None).comp
    else:
      arg = None
    call = building_blocks.Call(self._comp, arg)
    ref = _bind_computation_to_reference(
        call, 'calling a `federated_language.Value`'
    )
    return Value(ref)


def _dictlike_items_to_value(items, type_spec, container_type) -> Value:
  elements = []
  for i, (k, v) in enumerate(items):
    element_type = None if type_spec is None else type_spec[i]  # pytype: disable=unsupported-operands
    element_value = to_value(v, element_type)
    elements.append((k, element_value.comp))
  return Value(building_blocks.Struct(elements, container_type))


def to_value(
    arg: object,
    type_spec: Optional[computation_types.Type],
    *,
    parameter_type_hint=None,
    zip_if_needed: bool = False,
) -> Value:
  """Converts the argument into an instance of the abstract class `federated_language.Value`.

  Instances of `federated_language.Value` represent values that appear
  internally in federated computations. This helper function can be used to wrap
  a variety of Python objects as `federated_language.Value` instances to allow
  them to be passed as arguments, used as functions, or otherwise manipulated
  within bodies of federated computations.

  At the moment, the supported types include:

  * Simple constants of `str`, `int`, `float`, and `bool` types, mapped to
    values of a tensor type.

  * Numpy arrays (`np.ndarray` objects), also mapped to tensor type.

  * Dictionaries (`collections.OrderedDict` and unordered `dict`), `list`s,
    `tuple`s, `namedtuple`s, and `Struct`s, all of which are mapped to
    tuple type.

  * Computations (constructed with either the
    `federated_language.tensorflow.computation` or with the
    `federated_language.federated_computation` decorator), typically mapped to
    functions.

  * Placement literals (`federated_language.CLIENTS`,
    `federated_language.SERVER`), mapped to values of the placement type.

  This function is also invoked when attempting to execute a computation.
  All arguments supplied in the invocation are converted into values prior to
  execution. The types of Python objects that can be passed as arguments to
  computations thus matches the types listed here.

  Args:
    arg: An instance of one of the Python types that are convertible to values
      (instances of `federated_language.Value`).
    type_spec: An optional type specifier that allows for disambiguating the
      target type (e.g., when two types can be mapped to the same Python
      representations). If not specified, tried to determine the type of the
      value automatically.
    parameter_type_hint: An optional `federated_language.Type` or value
      convertible to it by `federated_language.to_type()` which specifies an
      argument type to use in the case that `arg` is a
      `polymorphic_computation.PolymorphicComputation`.
    zip_if_needed: If `True`, attempt to coerce the result of `to_value` to
      match `type_spec` by applying `intrinsics.federated_zip` to appropriate
      elements.

  Returns:
    An instance of `federated_language.Value` as described above.

  Raises:
    TypeError: if `arg` is of an unsupported type, or of a type that does not
      match `type_spec`. Raises explicit error message if backend constructs
      are encountered, as backend code should be sealed away from federated
      context.
  """
  # TODO: b/224484886 - Downcasting to all handled types.
  arg = typing.cast(
      Union[
          None,
          Value,
          building_blocks.ComputationBuildingBlock,
          placements.PlacementLiteral,
          computation_impl.ConcreteComputation,
          polymorphic_computation.PolymorphicComputation,
          computation_types.SequenceType,
          structure.Struct,
          py_typecheck.SupportsNamedTuple,
          Mapping[Hashable, object],
          tuple[object, ...],
          list[object],
          array.Array,
      ],
      arg,
  )
  if isinstance(arg, Value):
    result = arg
  elif isinstance(arg, building_blocks.ComputationBuildingBlock):
    result = Value(arg)
  elif isinstance(arg, placements.PlacementLiteral):
    result = Value(building_blocks.Placement(arg))
  elif isinstance(
      arg,
      (
          computation_impl.ConcreteComputation,
          polymorphic_computation.PolymorphicComputation,
      ),
  ):
    if isinstance(arg, polymorphic_computation.PolymorphicComputation):
      if parameter_type_hint is None:
        raise TypeError(
            'Polymorphic computations cannot be converted to'
            ' `federated_language.Value`s without a type hint. Consider'
            ' explicitly specifying the argument types of a computation before'
            ' passing it to a function that requires a'
            ' `federated_language.Value` (such as a intrinsic like'
            ' `federated_map`). If you are a developer and think this'
            ' should be supported, consider providing `parameter_type_hint` as'
            ' an argument to the encompassing `to_value` conversion.'
        )
      parameter_type_hint = computation_types.to_type(parameter_type_hint)
      arg = arg.fn_for_argument_type(parameter_type_hint)
    result = Value(arg.to_compiled_building_block())
  elif isinstance(arg, structure.Struct):
    items = structure.iter_elements(arg)
    result = _dictlike_items_to_value(items, type_spec, None)
  elif isinstance(arg, py_typecheck.SupportsNamedTuple):
    items = arg._asdict().items()
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif attrs.has(type(arg)):
    items = attrs.asdict(arg, recurse=False).items()
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif isinstance(arg, Mapping):
    result = _dictlike_items_to_value(arg.items(), type_spec, type(arg))
  elif isinstance(arg, (tuple, list)) and not isinstance(
      type_spec, computation_types.SequenceType
  ):
    items = zip(itertools.repeat(None), arg)
    result = _dictlike_items_to_value(items, type_spec, type(arg))
  elif isinstance(arg, typing.get_args(array.Array)):
    if type_spec is None:
      type_spec = type_conversions.infer_type(arg)
    if not isinstance(type_spec, computation_types.TensorType):
      raise ValueError(
          f'Expected a `federated_language.TensorType`, found {type_spec}.'
      )
    literal = building_blocks.Literal(arg, type_spec)
    result = Value(literal)
  else:
    raise TypeError(
        'Expected a Python types that is convertible to a'
        f' `federated_language.Value`, found {type(arg)}. If this is'
        ' backend-specific constructs, it was encountered in a federated'
        ' context and does not support mixing backend-specific and'
        ' federated logic. Please wrap any  backend-specific constructs in a'
        ' computation function.'
    )

  if type_spec is not None and not type_spec.is_assignable_from(
      result.type_signature
  ):
    if zip_if_needed:
      # Returns `None` if such a zip can't be performed.
      zipped_comp = building_block_factory.zip_to_match_type(
          comp_to_zip=result.comp, target_type=type_spec
      )
      if zipped_comp is not None:
        return Value(zipped_comp)
    raise computation_types.TypeNotAssignableError(
        type_spec, result.type_signature
    )
  return result
