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
"""A library of classes representing computations in a deserialized form."""

import abc
from collections.abc import Iterable, Iterator
import enum
import typing
from typing import Optional, Union
import zlib

from federated_language.common_libs import py_typecheck
from federated_language.common_libs import structure
from federated_language.compiler import array
from federated_language.compiler import intrinsic_defs
from federated_language.proto import computation_pb2
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_analysis
from federated_language.types import typed_object
import numpy as np

from google.protobuf import any_pb2


def _check_computation_has_field(
    computation_pb: computation_pb2.Computation, field: str
):
  if not computation_pb.HasField(field):
    raise ValueError(
        f'Expected `computation_pb` to have the field "{field}", found'
        f' {computation_pb}.'
    )


class UnexpectedBlockError(TypeError):

  def __init__(
      self,
      expected: type['ComputationBuildingBlock'],
      actual: 'ComputationBuildingBlock',
  ):
    message = f'Expected block of kind {expected}, found block {actual}'
    super().__init__(message)
    self.actual = actual
    self.expected = expected


class ComputationBuildingBlock(typed_object.TypedObject, abc.ABC):
  """The abstract base class for abstractions in the internal language.

  Instances of this class correspond roughly one-to-one to the abstractions
  defined in the `Computation` message in the `computation.proto`, and are
  intended primarily for the ease of manipulating the abstract syntax trees
  (AST) of federated computations as they are transformed by the compiler
  pipeline to mold into the needs of a particular execution backend. The only
  abstraction that does not have a dedicated Python equivalent is a section
  of backend-specific code (it's represented by
  `federated_language.framework.CompiledComputation`).
  """

  def __init__(self, type_signature):
    """Constructs a computation building block with the given type.

    Args:
      type_signature: An instance of types.Type, or something convertible to it
        via types.to_type().
    """
    self._type_signature = computation_types.to_type(type_signature)
    self._hash = None
    self._proto = None

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'ComputationBuildingBlock':
    """Returns a `ComputationBuildingBlock` for the `computation_pb`."""
    computation_oneof = computation_pb.WhichOneof('computation')
    if computation_oneof == 'block':
      return Block.from_proto(computation_pb)
    elif computation_oneof == 'call':
      return Call.from_proto(computation_pb)
    elif computation_oneof == 'data':
      return Data.from_proto(computation_pb)
    elif computation_oneof == 'intrinsic':
      return Intrinsic.from_proto(computation_pb)
    elif computation_oneof == 'lambda':
      return Lambda.from_proto(computation_pb)
    elif computation_oneof == 'literal':
      return Literal.from_proto(computation_pb)
    elif computation_oneof == 'placement':
      return Placement.from_proto(computation_pb)
    elif computation_oneof == 'reference':
      return Reference.from_proto(computation_pb)
    elif computation_oneof == 'selection':
      return Selection.from_proto(computation_pb)
    elif computation_oneof == 'struct':
      return Struct.from_proto(computation_pb)
    elif computation_oneof == 'tensorflow':
      return CompiledComputation.from_proto(computation_pb)
    elif computation_oneof == 'xla':
      return CompiledComputation.from_proto(computation_pb)
    else:
      raise NotImplementedError(
          f'Unexpected computation found: {computation_oneof}.'
      )

  @abc.abstractmethod
  def to_proto(self):
    """Returns a `computation_pb2.Computation` for this building block."""
    raise NotImplementedError

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  @abc.abstractmethod
  def children(self) -> Iterator['ComputationBuildingBlock']:
    """Returns an iterator yielding immediate child building blocks."""
    raise NotImplementedError

  def compact_representation(self):
    """Returns the compact string representation of this building block."""
    return _string_representation(self, formatted=False)

  def formatted_representation(self):
    """Returns the formatted string representation of this building block."""
    return _string_representation(self, formatted=True)

  def structural_representation(self):
    """Returns the structural string representation of this building block."""
    return _structural_representation(self)

  def __str__(self):
    return self.compact_representation()

  @abc.abstractmethod
  def __repr__(self):
    raise NotImplementedError


class Reference(ComputationBuildingBlock):
  """A reference to a name defined earlier in the internal language.

  Names are defined by lambda expressions (which have formal named parameters),
  and block structures (which can have one or more locals). The reference
  construct is used to refer to those parameters or locals by a string name.
  The usual hiding rules apply. A reference binds to the closest definition of
  the given name in the most deeply nested surrounding lambda or block.

  A concise notation for a reference to name `foo` is `foo`. For example, in
  a lambda expression `(x -> f(x))` there are two references, one to `x` that
  is defined as the formal parameter of the lambda epxression, and one to `f`
  that must have been defined somewhere in the surrounding context.
  """

  def __init__(self, name: str, type_signature: object, context=None):
    """Creates a reference to 'name' of type 'type_signature' in context 'context'.

    Args:
      name: The name of the referenced entity.
      type_signature: The type spec of the referenced entity.
      context: The optional context in which the referenced entity is defined.
        This class does not prescribe what Python type the 'context' needs to be
        and merely exposes it as a property (see below). The only requirement is
        that the context implements str() and repr().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    super().__init__(type_signature)
    self._name = name
    self._context = context

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'Reference':
    """Returns a `Reference` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'reference')

    type_signature = computation_types.Type.from_proto(computation_pb.type)
    return Reference(computation_pb.reference.name, type_signature)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      reference_pb = computation_pb2.Reference(name=self._name)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          reference=reference_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def name(self) -> str:
    return self._name

  @property
  def context(self):
    return self._context

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Reference):
      return NotImplemented
    # IMPORTANT: References are only equal to each other if they are the same
    # object because two references with the same `name` are different if they
    # are in different locations within the same scope, in different scopes, or
    # in different contexts.
    return False

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._name, self._type_signature))
    return self._hash

  def __repr__(self):
    if self._context is not None:
      return (
          f'Reference({self._name!r}, {self._type_signature!r},'
          f' {self._context!r})'
      )
    else:
      return f'Reference({self._name!r}, {self._type_signature!r})'


class Selection(ComputationBuildingBlock):
  """A selection by name or index from a struct-typed value in the language.

  The concise syntax for selections is `foo.bar` (selecting a named `bar` from
  the value of expression `foo`), and `foo[n]` (selecting element at index `n`
  from the value of `foo`).
  """

  def __init__(
      self,
      source: ComputationBuildingBlock,
      name: Optional[str] = None,
      index: Optional[int] = None,
  ):
    """A selection from 'source' by a string or numeric 'name_or_index'.

    Exactly one of 'name' or 'index' must be specified (not None).

    Args:
      source: The source value to select from (an instance of
        ComputationBuildingBlock).
      name: A string name of the element to be selected.
      index: A numeric index of the element to be selected.

    Raises:
      TypeError: if arguments are of the wrong types.
      ValueError: if the name is empty or index is negative, or the name/index
        is not compatible with the type signature of the source, or neither or
        both are defined (not None).
    """
    source_type = source.type_signature
    # TODO: b/224484886 - Downcasting to all handled types.
    source_type = typing.cast(Union[computation_types.StructType], source_type)
    if not isinstance(source_type, computation_types.StructType):
      raise TypeError(
          'Expected the source of selection to be a `StructType`, '
          'instead found it to be of type {}.'.format(source_type)
      )
    if name is not None and index is not None:
      raise ValueError(
          'Cannot simultaneously specify a name and an index, choose one.'
      )
    if name is not None:
      if not name:
        raise ValueError('The name of the selected element cannot be empty.')
      # Normalize, in case we are dealing with a Unicode type or some such.
      name = str(name)
      if name not in source_type.fields():
        raise ValueError(
            f'Error selecting named field `{name}` from type `{source_type}`, '
            f'whose only named fields are {source_type.fields()}.'
        )
      type_signature = source_type[name]
    elif index is not None:
      length = len(source_type)
      if index < 0 or index >= length:
        raise ValueError(
            f'The index `{index}` does not fit into the valid range in the '
            f'struct type: 0..{length}'
        )
      type_signature = source_type[index]
    else:
      raise ValueError(
          'Must define either a name or index, and neither was specified.'
      )
    super().__init__(type_signature)
    self._source = source
    self._name = name
    self._index = index

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'Selection':
    """Returns a `Selection` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'selection')

    source = ComputationBuildingBlock.from_proto(
        computation_pb.selection.source
    )
    return Selection(source, index=computation_pb.selection.index)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      source_pb = self._source.to_proto()
      selection_pb = computation_pb2.Selection(
          source=source_pb,
          index=self.as_index(),
      )
      return computation_pb2.Computation(
          type=type_pb,
          selection=selection_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._source

  @property
  def source(self) -> ComputationBuildingBlock:
    return self._source

  @property
  def name(self) -> Optional[str]:
    return self._name

  @property
  def index(self) -> Optional[int]:
    return self._index

  def as_index(self) -> int:
    if self._index is not None:
      return self._index
    else:
      names = [n for n, _ in self._source.type_signature.items()]  # pytype: disable=attribute-error
      return names.index(self._name)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Selection):
      return NotImplemented
    return (
        self._source,
        self._name,
        self._index,
    ) == (
        other._source,
        other._name,
        other._index,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._source, self._name, self._index))
    return self._hash

  def __repr__(self):
    if self._name is not None:
      return f'Selection({self._source!r}, name={self._name!r})'
    else:
      return f'Selection({self._source!r}, index={self._index!r})'


class Struct(ComputationBuildingBlock, structure.Struct):
  """A struct with named or unnamed elements in the internal language.

  The concise notation for structs is `<name_1=value_1, ...., name_n=value_n>`
  for structs with named elements, `<value_1, ..., value_n>` for structs with
  unnamed elements, or a mixture of these for structs with some named and some
  unnamed elements, where `name_k` are the names, and `value_k` are the value
  expressions.

  For example, a lambda expression that applies `fn` to elements of 2-structs
  pointwise could be represented as `(arg -> <fn(arg[0]),fn(arg[1])>)`.
  """

  def __init__(self, elements, container_type=None):
    """Constructs a struct from the given list of elements.

    Args:
      elements: The elements of the struct, supplied as a list of (name, value)
        pairs, where 'name' can be None in case the corresponding element is not
        named and only accessible via an index (see also `structure.Struct`).
      container_type: An optional Python container type to associate with the
        struct.

    Raises:
      TypeError: if arguments are of the wrong types.
    """

    # Not using super() here and below, as the two base classes have different
    # signatures of their constructors, and the struct implementation
    # of selection interfaces should override that in the generic class 'Value'
    # to favor simplified expressions where simplification is possible.
    def _map_element(e):
      """Returns a named or unnamed element."""
      if isinstance(e, ComputationBuildingBlock):
        return (None, e)
      elif py_typecheck.is_name_value_pair(
          e, value_type=ComputationBuildingBlock
      ):
        if e[0] is not None and not e[0]:
          raise ValueError('Unexpected struct element with empty string name.')
        return (e[0], e[1])
      else:
        raise TypeError('Unexpected struct element: {}.'.format(e))

    elements = [_map_element(e) for e in elements]
    element_pairs = [
        ((e[0], e[1].type_signature) if e[0] else e[1].type_signature)
        for e in elements
    ]

    if container_type is None:
      type_signature = computation_types.StructType(element_pairs)
    else:
      type_signature = computation_types.StructWithPythonType(
          element_pairs, container_type
      )
    ComputationBuildingBlock.__init__(self, type_signature)
    structure.Struct.__init__(self, elements)
    self._type_signature = type_signature

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Struct':
    """Returns a `Struct` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'struct')
    elements = []
    for element_pb in computation_pb.struct.element:
      if element_pb.name:
        name = element_pb.name
      else:
        name = None
      element = ComputationBuildingBlock.from_proto(element_pb.value)
      elements.append((name, element))
    return Struct(elements)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      element_pbs = []
      for name, element in self.items():
        value_pb = element.to_proto()
        element_pb = computation_pb2.Struct.Element(
            name=name,
            value=value_pb,
        )
        element_pbs.append(element_pb)
      struct_pb = computation_pb2.Struct(element=element_pbs)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          struct=struct_pb,
      )
    return self._proto

  @property
  def type_signature(self) -> computation_types.StructType:
    return self._type_signature

  def children(self) -> Iterator[ComputationBuildingBlock]:
    return (element for _, element in structure.iter_elements(self))

  def items(self) -> Iterator[tuple[Optional[str], ComputationBuildingBlock]]:
    return structure.iter_elements(self)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Struct):
      return NotImplemented
    if self._type_signature != other._type_signature:
      return False
    return structure.Struct.__eq__(self, other)

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((
          structure.Struct.__hash__(self),
          self._type_signature,
      ))
    return self._hash

  def __repr__(self):
    elements = list(self.items())
    return f'Struct({elements!r})'


class Call(ComputationBuildingBlock):
  """A representation of a function invocation in the internal language.

  The call construct takes an argument struct with two elements, the first being
  the function to invoke (represented as a computation with a functional result
  type), and the second being the argument to feed to that function. Typically,
  the function is either an intrinsic, or a lambda expression.

  The concise notation for calls is `foo(bar)`, where `foo` is the function,
  and `bar` is the argument.
  """

  def __init__(
      self,
      fn: ComputationBuildingBlock,
      arg: Optional[ComputationBuildingBlock] = None,
  ):
    """Creates a call to 'fn' with argument 'arg'.

    Args:
      fn: A value of a functional type that represents the function to invoke.
      arg: The optional argument, present iff 'fn' expects one, of a type that
        matches the type of 'fn'.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    function_type = fn.type_signature
    # TODO: b/224484886 - Downcasting to all handled types.
    function_type = typing.cast(
        Union[computation_types.FunctionType], function_type
    )
    if not isinstance(function_type, computation_types.FunctionType):
      raise TypeError(
          'Expected `fn` to have a `federated_language.FunctionType`, found'
          f' {function_type}.'
      )
    parameter_type = function_type.parameter
    if parameter_type is not None:
      if arg is None:
        raise TypeError(
            f'Expected `arg` to be of type {parameter_type}, found None.'
        )
      elif not parameter_type.is_assignable_from(arg.type_signature):
        raise TypeError(
            f'Expected `arg` to be of type {parameter_type}, found an'
            f' incompatible type {arg.type_signature}.'
        )
    else:
      if arg is not None:
        raise TypeError(f'Expected `arg` to be None, found {arg}.')
    super().__init__(function_type.result)
    self._function = fn
    self._argument = arg

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Call':
    """Returns a `Call` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'call')

    fn = ComputationBuildingBlock.from_proto(computation_pb.call.function)
    arg_proto = computation_pb.call.argument
    if arg_proto.WhichOneof('computation') is not None:
      arg = ComputationBuildingBlock.from_proto(arg_proto)
    else:
      arg = None
    return Call(fn, arg)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      function_pb = self._function.to_proto()
      if self._argument is not None:
        argument_pb = self._argument.to_proto()
      else:
        argument_pb = None
      call_pb = computation_pb2.Call(
          function=function_pb,
          argument=argument_pb,
      )
      self._proto = computation_pb2.Computation(
          type=type_pb,
          call=call_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._function
    if self._argument is not None:
      yield self._argument

  @property
  def function(self) -> ComputationBuildingBlock:
    return self._function

  @property
  def argument(self) -> Optional[ComputationBuildingBlock]:
    return self._argument

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Call):
      return NotImplemented
    return (
        self._function,
        self._argument,
    ) == (
        other._function,
        other._argument,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._function, self._argument))
    return self._hash

  def __repr__(self):
    if self._argument is not None:
      return f'Call({self._function!r}, {self._argument!r})'
    else:
      return f'Call({self._function!r})'


class Lambda(ComputationBuildingBlock):
  """A representation of a lambda expression in the internal language.

  A lambda expression consists of a string formal parameter name, and a result
  expression that can contain references by name to that formal parameter. A
  concise notation for lambdas is `(foo -> bar)`, where `foo` is the name of
  the formal parameter, and `bar` is the result expression.
  """

  def __init__(
      self,
      parameter_name: Optional[str],
      parameter_type: Optional[object],
      result: ComputationBuildingBlock,
  ):
    """Creates a lambda expression.

    Args:
      parameter_name: The (string) name of the parameter accepted by the lambda.
        This name can be used by Reference() instances in the body of the lambda
        to refer to the parameter. Note that an empty parameter name shall be
        treated as equivalent to no parameter.
      parameter_type: The type of the parameter, an instance of types.Type or
        something convertible to it by types.to_type().
      result: The resulting value produced by the expression that forms the body
        of the lambda. Must be an instance of ComputationBuildingBlock.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    if not parameter_name:
      parameter_name = None
    if (parameter_name is None) != (parameter_type is None):
      raise TypeError(
          'A lambda expression must have either a valid parameter name and '
          'type or both parameter name and type must be `None`. '
          '`parameter_name` was {} but `parameter_type` was {}.'.format(
              parameter_name, parameter_type
          )
      )
    if parameter_name is not None:
      parameter_type = computation_types.to_type(parameter_type)
    type_signature = computation_types.FunctionType(
        parameter_type, result.type_signature
    )
    super().__init__(type_signature)
    self._parameter_name = parameter_name
    self._parameter_type = parameter_type
    self._result = result
    self._type_signature = type_signature

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Lambda':
    """Returns a `Lambda` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'lambda')

    fn_pb: computation_pb2.Lambda = getattr(computation_pb, 'lambda')
    if computation_pb.type.function.HasField('parameter'):
      parameter_type = computation_types.Type.from_proto(
          computation_pb.type.function.parameter
      )
    else:
      parameter_type = None
    result = ComputationBuildingBlock.from_proto(fn_pb.result)
    return Lambda(fn_pb.parameter_name, parameter_type, result)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      result_pb = self._result.to_proto()
      fn_pb = computation_pb2.Lambda(
          parameter_name=self._parameter_name,
          result=result_pb,
      )
      # We are unpacking the lambda argument here because `lambda` is a reserved
      # keyword in Python, but it is also the name of the parameter for a
      # `computation_pb2.Computation`.
      self._proto = computation_pb2.Computation(
          type=type_pb, **{'lambda': fn_pb}
      )
    return self._proto

  @property
  def type_signature(self) -> computation_types.FunctionType:
    return self._type_signature

  def children(self) -> Iterator[ComputationBuildingBlock]:
    yield self._result

  @property
  def parameter_name(self) -> Optional[str]:
    return self._parameter_name

  @property
  def parameter_type(self) -> Optional[computation_types.Type]:
    return self._parameter_type

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Lambda):
      return NotImplemented
    return (
        self._parameter_name,
        self._parameter_type,
        self._result,
    ) == (
        other._parameter_name,
        other._parameter_type,
        other._result,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((
          self._parameter_name,
          self._parameter_type,
          self._result,
      ))
    return self._hash

  def __repr__(self) -> str:
    return (
        f'Lambda({self._parameter_name!r}, {self._parameter_type!r},'
        f' {self._result!r})'
    )


class Block(ComputationBuildingBlock):
  """A representation of a block of code in the internal language.

  A block is a syntactic structure that consists of a sequence of local name
  bindings followed by a result. The bindings are interpreted sequentially,
  with bindings later in the sequence in the scope of those listed earlier,
  and the result in the scope of the entire sequence. The usual hiding rules
  apply.

  An informal concise notation for blocks is the following, with `name_k`
  representing the names defined locally for the block, `value_k` the values
  associated with them, and `result` being the expression that reprsents the
  value of the block construct.

  ```
  let name_1=value_1, name_2=value_2, ..., name_n=value_n in result
  ```

  Blocks are technically a redundant abstraction, as they can be equally well
  represented by lambda expressions. A block of the form `let x=y in z` is
  roughly equivalent to `(x -> z)(y)`. Although redundant, blocks have a use
  as a way to reduce computation ASTs to a simpler, less nested and more
  readable form, and are helpful in AST transformations as a mechanism that
  prevents possible naming conflicts.

  An example use of a block expression to flatten a nested structure below:

  ```
  z = federated_sum(federated_map(x, federated_broadcast(y)))
  ```

  An equivalent form in a more sequential notation using a block expression:
  ```
  let
    v1 = federated_broadcast(y),
    v2 = federated_map(x, v1)
  in
    federated_sum(v2)
  ```
  """

  def __init__(
      self,
      local_symbols: Iterable[tuple[str, ComputationBuildingBlock]],
      result: ComputationBuildingBlock,
  ):
    """Creates a block of federated_language code.

    Args:
      local_symbols: The list of one or more local declarations, each of which
        is a 2-tuple (name, value), with 'name' being the string name of a local
        symbol being defined, and 'value' being the instance of
        ComputationBuildingBlock, the output of which will be locally bound to
        that name.
      result: An instance of ComputationBuildingBlock that computes the result.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    updated_locals = []
    for index, element in enumerate(local_symbols):
      if (
          not isinstance(element, tuple)
          or (len(element) != 2)
          or not isinstance(element[0], str)
      ):
        raise TypeError(
            'Expected the locals to be a list of 2-element structs with string '
            'name as their first element, but this is not the case for the '
            'local at position {} in the sequence: {}.'.format(index, element)
        )
      name = element[0]
      value = element[1]
      updated_locals.append((name, value))
    super().__init__(result.type_signature)
    self._local_symbols = updated_locals
    self._result = result

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Block':
    """Returns a `Block` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'block')

    local_symbols = []
    for local_symbol_pb in computation_pb.block.local:
      if local_symbol_pb.name:
        name = local_symbol_pb.name
      else:
        name = None
      symbol = ComputationBuildingBlock.from_proto(local_symbol_pb.value)
      local_symbols.append((name, symbol))
    result = ComputationBuildingBlock.from_proto(computation_pb.block.result)
    return Block(local_symbols, result)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()

      local_symbol_pbs = []
      for name, local_symbol in self._local_symbols:
        value_pb = local_symbol.to_proto()
        local_symbol_pb = computation_pb2.Block.Local(
            name=name,
            value=value_pb,
        )
        local_symbol_pbs.append(local_symbol_pb)
      result_pb = self._result.to_proto()
      block_pb = computation_pb2.Block(
          **{'local': local_symbol_pbs},
          result=result_pb,
      )
      self._proto = computation_pb2.Computation(
          type=type_pb,
          block=block_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    for _, value in self._local_symbols:
      yield value
    yield self._result

  @property
  def locals(self) -> list[tuple[str, ComputationBuildingBlock]]:
    return list(self._local_symbols)

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Block):
      return NotImplemented
    return (
        self._local_symbols,
        self._result,
    ) == (
        other._local_symbols,
        other._result,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((tuple(self._local_symbols), self._result))
    return self._hash

  def __repr__(self) -> str:
    return f'Block({self._local_symbols!r}, {self._result!r})'


class Intrinsic(ComputationBuildingBlock):
  """A representation of an intrinsic in the internal language.

  An intrinsic is a symbol known to the compiler pipeline, represented
  as a known URI. It generally appears in expressions with a concrete type,
  although all intrinsic are defined with template types. This class does not
  deal with parsing intrinsic URIs and verifying their types, it is only a
  container. Parsing and type analysis are a responsibility of the components
  that manipulate ASTs. See intrinsic_defs.py for the list of known intrinsics.
  """

  def __init__(self, uri: str, type_signature: computation_types.Type):
    """Creates an intrinsic.

    Args:
      uri: The URI of the intrinsic.
      type_signature: A `federated_language.Type`, the type of the intrinsic.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(uri)
    if intrinsic_def is not None:
      # NOTE: this is really expensive.
      type_analysis.check_concrete_instance_of(
          type_signature, intrinsic_def.type_signature
      )
    super().__init__(type_signature)
    self._uri = uri

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'Intrinsic':
    """Returns a `Intrinsic` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'intrinsic')

    type_signature = computation_types.Type.from_proto(computation_pb.type)
    return Intrinsic(computation_pb.intrinsic.uri, type_signature)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      intrinsic_pb = computation_pb2.Intrinsic(uri=self._uri)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          intrinsic=intrinsic_pb,
      )
    return self._proto

  def intrinsic_def(self) -> intrinsic_defs.IntrinsicDef:
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(self._uri)
    if intrinsic_def is None:
      raise ValueError(
          'Failed to retrieve definition of intrinsic with URI '
          f'`{self._uri}`. Perhaps a definition needs to be added to '
          '`intrinsic_defs.py`?'
      )
    return intrinsic_def

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def uri(self) -> str:
    return self._uri

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Intrinsic):
      return NotImplemented
    return (
        self._uri,
        self._type_signature,
    ) == (
        other._uri,
        other._type_signature,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._uri, self._type_signature))
    return self._hash

  def __repr__(self) -> str:
    return f'Intrinsic({self._uri!r}, {self._type_signature!r})'


class Data(ComputationBuildingBlock):
  """A representation of data (an input pipeline).

  This class does not deal with parsing data protos and verifying correctness,
  it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  def __init__(self, content: any_pb2.Any, type_signature: object):
    """Creates a representation of data.

    Args:
      content: The proto that characterizes the data.
      type_signature: Either the types.Type that represents the type of this
        data, or something convertible to it by types.to_type().
    """
    super().__init__(type_signature)
    self._content = content

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Data':
    """Returns a `Data` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'data')

    type_signature = computation_types.Type.from_proto(computation_pb.type)
    return Data(computation_pb.data.content, type_signature)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      data_pb = computation_pb2.Data(content=self._content)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          data=data_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def content(self) -> any_pb2.Any:
    return self._content

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Data):
      return NotImplemented
    return (
        self._content,
        self._type_signature,
    ) == (
        other._content,
        other._type_signature,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((str(self._content), self._type_signature))
    return self._hash

  def __repr__(self) -> str:
    return f'Data({self._content!r}, {self._type_signature!r})'


class CompiledComputation(ComputationBuildingBlock):
  """A representation of a fully constructed and serialized computation.

  A compiled computation is one that has not been parsed into constituents, and
  is simply represented as an embedded `Computation` protocol buffer. Whereas
  technically, any computation can be represented and passed around this way,
  this structure is generally only used to represent TensorFlow sections, for
  which otherwise there isn't any dedicated structure.
  """

  def __init__(
      self,
      proto: computation_pb2.Computation,
      name: Optional[str] = None,
      type_signature: Optional[computation_types.Type] = None,
  ):
    """Creates a representation of a fully constructed computation.

    Args:
      proto: An instance of computation_pb2.Computation with the computation
        logic.
      name: An optional string name to associate with this computation, used
        only for debugging purposes. If the name is not specified (None), it is
        autogenerated as a hexadecimal string from the hash of the proto.
      type_signature: An optional type signature to associate with this
        computation rather than the serialized one.
    """
    if type_signature is None:
      type_signature = computation_types.Type.from_proto(proto.type)
    super().__init__(type_signature)
    self._proto = proto
    if name is None:
      name = '{:x}'.format(zlib.adler32(self._proto.SerializeToString()))
    self._name = name

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'CompiledComputation':
    """Returns a `CompiledComputation` for the `computation_pb`."""
    return CompiledComputation(computation_pb)

  def to_proto(self) -> computation_pb2.Computation:
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def name(self) -> str:
    return self._name

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, CompiledComputation):
      return NotImplemented
    return (
        self._proto,
        self._name,
        self._type_signature,
    ) == (
        other._proto,
        other._name,
        other._type_signature,
    )

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((
          self._proto.SerializeToString(),
          self._name,
          self._type_signature,
      ))
    return self._hash

  def __repr__(self) -> str:
    return f'CompiledComputation({self._name!r}, {self._type_signature!r})'


class Placement(ComputationBuildingBlock):
  """A representation of a placement literal in the internal language.

  Currently this can only be `federated_language.SERVER` or
  `federated_language.CLIENTS`.
  """

  def __init__(self, literal: placements.PlacementLiteral):
    """Constructs a new placement instance for the given placement literal.

    Args:
      literal: The placement literal.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    super().__init__(computation_types.PlacementType())
    self._literal = literal

  @classmethod
  def from_proto(
      cls, computation_pb: computation_pb2.Computation
  ) -> 'Placement':
    """Returns a `Placement` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'placement')

    literal = placements.uri_to_placement_literal(computation_pb.placement.uri)
    return Placement(literal)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      placement_pb = computation_pb2.Placement(uri=self._literal.uri)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          placement=placement_pb,
      )
    return self._proto

  def children(self) -> Iterator[ComputationBuildingBlock]:
    del self
    return iter(())

  @property
  def uri(self) -> str:
    return self._literal.uri

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Placement):
      return NotImplemented
    return self._literal == other._literal

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._literal))
    return self._hash

  def __repr__(self) -> str:
    return f'Placement({self._literal.uri!r})'


class Literal(ComputationBuildingBlock):
  """A representation of a literal in the internal language."""

  def __init__(
      self, value: array.Array, type_signature: computation_types.TensorType
  ):
    """Returns an initialized `federated_language.framework.Literal`.

    Args:
      value: The value of the literal.
      type_signature: A `federated_language.TensorType`.

    Raises:
      ValueError: If `value` is not compatible with `type_signature`.
    """
    if (
        isinstance(value, (np.generic, np.ndarray))
        and value.dtype.type is np.str_
    ):
      value = value.astype(np.bytes_)
    elif isinstance(value, str):
      value = value.encode()

    if not array.is_compatible_dtype(value, type_signature.dtype.type):
      raise ValueError(
          f"Expected '{value}' to be compatible with"
          f" '{type_signature.dtype.type}'."
      )

    if not array.is_compatible_shape(value, type_signature.shape):
      raise ValueError(
          f"Expected '{value}' to be compatible with '{type_signature.shape}'."
      )

    super().__init__(type_signature)
    self._value = value
    self._type_signature = type_signature
    self._hash = None

  @classmethod
  def from_proto(cls, computation_pb: computation_pb2.Computation) -> 'Literal':
    """Returns a `Literal` for the `computation_pb`."""
    _check_computation_has_field(computation_pb, 'literal')

    value = array.from_proto(computation_pb.literal.value)
    type_signature = computation_types.Type.from_proto(computation_pb.type)
    if not isinstance(type_signature, computation_types.TensorType):
      raise ValueError(
          'Expected `type_signature` to be a `federated_language.TensorType`,'
          f' found {type(type_signature)}.'
      )
    return Literal(value, type_signature)

  def to_proto(self) -> computation_pb2.Computation:
    """Returns a `computation_pb2.Computation` for this building block."""
    if self._proto is None:
      type_pb = self._type_signature.to_proto()
      value_pb = array.to_proto(
          self._value, dtype_hint=self.type_signature.dtype.type
      )
      literal_pb = computation_pb2.Literal(value=value_pb)
      self._proto = computation_pb2.Computation(
          type=type_pb,
          literal=literal_pb,
      )
    return self._proto

  @property
  def type_signature(self) -> computation_types.TensorType:
    return self._type_signature

  def children(self) -> Iterator[ComputationBuildingBlock]:
    return iter(())

  @property
  def value(self) -> object:
    return self._value

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Literal):
      return NotImplemented

    if self._type_signature != other._type_signature:
      return False
    if isinstance(self._value, np.ndarray) and isinstance(
        other._value, np.ndarray
    ):
      return np.array_equal(self._value, other._value)
    else:
      return self._value == other._value

  def __hash__(self):
    if self._hash is None:
      if isinstance(self._value, (np.generic, np.ndarray)):
        hashable_value = tuple(self._value.flatten().tolist())
      else:
        hashable_value = self._value
      self._hash = hash((hashable_value, self._type_signature))
    return self._hash

  def __repr__(self) -> str:
    if isinstance(self._value, np.ndarray):
      value_repr = (
          f'np.array({self._value.tolist()},'
          f' dtype=np.{self._value.dtype.type.__name__})'
      )
    else:
      value_repr = repr(self._value)
    return f'Literal({value_repr}, {self._type_signature!r})'


def _string_representation(
    comp: ComputationBuildingBlock,
    formatted: bool,
) -> str:
  """Returns the string representation of a `ComputationBuildingBlock`.

  This functions creates a `list` of strings representing the given `comp`;
  combines the strings in either a formatted or un-formatted representation; and
  returns the resulting string represetnation.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.
    formatted: A boolean indicating if the returned string should be formatted.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """

  def _join(components: Iterable[list[str]]) -> list[str]:
    """Returns a `list` of strings by combining each component in `components`.

    >>> _join([['a'], ['b'], ['c']])
    ['abc']

    >>> _join([['a', 'b', 'c'], ['d', 'e', 'f']])
    ['abcd', 'ef']

    This function is used to help track where new-lines should be inserted into
    the string representation if the lines are formatted.

    Args:
      components: A `list` where each element is a `list` of strings
        representing a part of the string of a `ComputationBuildingBlock`.
    """
    lines = ['']
    for component in components:
      lines[-1] = '{}{}'.format(lines[-1], component[0])
      lines.extend(component[1:])
    return lines

  def _indent(lines, indent_chars='  '):
    """Returns a `list` of strings indented across a slice."""
    return ['{}{}'.format(indent_chars, e) for e in lines]

  def _lines_for_named_comps(named_comps, formatted):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named computations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    lines = []
    for index, (name, comp) in enumerate(named_comps):
      if index != 0:
        if formatted:
          lines.append([',', ''])
        else:
          lines.append([','])
      element_lines = _lines_for_comp(comp, formatted)
      if name is not None:
        element_lines = _join([
            ['{}='.format(name)],
            element_lines,
        ])
      lines.append(element_lines)
    return _join(lines)

  def _lines_for_comp(comp, formatted):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    if isinstance(comp, Block):
      lines = []
      variables_lines = _lines_for_named_comps(comp.locals, formatted)
      if formatted:
        variables_lines = _indent(variables_lines)
        lines.extend([['(let ', ''], variables_lines, ['', ' in ']])
      else:
        lines.extend([['(let '], variables_lines, [' in ']])
      result_lines = _lines_for_comp(comp.result, formatted)
      lines.append(result_lines)
      lines.append([')'])
      return _join(lines)
    elif isinstance(comp, Reference):
      if comp.context is not None:
        return ['{}@{}'.format(comp.name, comp.context)]
      else:
        return [comp.name]
    elif isinstance(comp, Selection):
      source_lines = _lines_for_comp(comp.source, formatted)
      if comp.name is not None:
        return _join([source_lines, ['.{}'.format(comp.name)]])
      else:
        return _join([source_lines, ['[{}]'.format(comp.index)]])
    elif isinstance(comp, Call):
      function_lines = _lines_for_comp(comp.function, formatted)
      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument, formatted)
        return _join([function_lines, ['('], argument_lines, [')']])
      else:
        return _join([function_lines, ['()']])
    elif isinstance(comp, CompiledComputation):
      return ['comp#{}'.format(comp.name)]
    elif isinstance(comp, Data):
      return [str(id(comp.content))]
    elif isinstance(comp, Intrinsic):
      return [comp.uri]
    elif isinstance(comp, Lambda):
      result_lines = _lines_for_comp(comp.result, formatted)
      if comp.parameter_type is None:
        param_name = ''
      else:
        param_name = comp.parameter_name
      lines = [['({} -> '.format(param_name)], result_lines, [')']]
      return _join(lines)
    elif isinstance(comp, Placement):
      placement_literal = placements.uri_to_placement_literal(comp.uri)
      return [placement_literal.name]
    elif isinstance(comp, Literal):
      return [str(comp.value)]
    elif isinstance(comp, Struct):
      if not comp:
        return ['<>']
      elements = comp.items()
      elements_lines = _lines_for_named_comps(elements, formatted)
      if formatted:
        elements_lines = _indent(elements_lines)
        lines = [['<', ''], elements_lines, ['', '>']]
      else:
        lines = [['<'], elements_lines, ['>']]
      return _join(lines)
    else:
      raise NotImplementedError(
          f'Unexpected building block found: {type(comp)}.'
      )

  lines = _lines_for_comp(comp, formatted)
  lines = [line.rstrip() for line in lines]
  if formatted:
    return '\n'.join(lines)
  else:
    return ''.join(lines)


def _structural_representation(comp: ComputationBuildingBlock):
  """Returns the structural string representation of the given `comp`.

  This functions creates and returns a string representing the structure of the
  abstract syntax tree for the given `comp`.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """
  padding_char = ' '

  def _get_leading_padding(string):
    """Returns the length of the leading padding for the given `string`."""
    for index, character in enumerate(string):
      if character != padding_char:
        return index
    return len(string)

  def _get_trailing_padding(string):
    """Returns the length of the trailing padding for the given `string`."""
    for index, character in enumerate(reversed(string)):
      if character != padding_char:
        return index
    return len(string)

  def _pad_left(lines, total_width):
    """Pads the beginning of each line in `lines` to the given `total_width`.

    >>>_pad_left(['aa', 'bb'], 4)
    ['  aa', '  bb',]

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_left(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(padding_char * padding, line)

    return [_pad_line_left(line, total_width) for line in lines]

  def _pad_right(lines, total_width):
    """Pads the end of each line in `lines` to the given `total_width`.

    >>>_pad_right(['aa', 'bb'], 4)
    ['aa  ', 'bb  ']

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_right(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(line, padding_char * padding)

    return [_pad_line_right(line, total_width) for line in lines]

  class Alignment(enum.Enum):
    LEFT = 1
    RIGHT = 2

  def _concatenate(lines_1, lines_2, align):
    """Concatenates two `list`s of strings.

    Concatenates two `list`s of strings by appending one list of strings to the
    other and then aligning lines of different widths by either padding the left
    or padding the right of each line to the width of the longest line.

    >>>_concatenate(['aa', 'bb'], ['ccc'], Alignment.LEFT)
    ['aa ', 'bb ', 'ccc']

    Args:
      lines_1: A `list` of strings.
      lines_2: A `list` of strings.
      align: An enum indicating how to align lines of different widths.

    Returns:
      A `list` of lines.
    """
    lines = lines_1 + lines_2
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    if align is Alignment.LEFT:
      return _pad_right(lines, longest_width)
    elif align is Alignment.RIGHT:
      return _pad_left(lines, longest_width)

  def _calculate_inset_from_padding(
      left, right, preferred_padding, minimum_content_padding
  ):
    """Calculates the inset for the given padding.

    NOTE: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      An integer.
    """
    assert preferred_padding >= 0
    assert minimum_content_padding >= 0

    trailing_padding = _get_trailing_padding(left[0])
    leading_padding = _get_leading_padding(right[0])
    inset = trailing_padding + leading_padding - preferred_padding
    for left_line, right_line in zip(left[1:], right[1:]):
      trailing_padding = _get_trailing_padding(left_line)
      leading_padding = _get_leading_padding(right_line)
      minimum_inset = (
          trailing_padding + leading_padding - minimum_content_padding
      )
      inset = min(inset, minimum_inset)
    return inset

  def _fit_with_inset(left, right, inset):
    r"""Concatenates the lines of two `list`s of strings.

    NOTE: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      inset: The amount of padding to remove or add when concatenating the
        lines.

    Returns:
      A `list` of lines.
    """
    lines = []
    for left_line, right_line in zip(left, right):
      if inset > 0:
        left_inset = 0
        trailing_padding = _get_trailing_padding(left_line)
        if trailing_padding > 0:
          left_inset = min(trailing_padding, inset)
          left_line = left_line[:-left_inset]
        if inset - left_inset > 0:
          leading_padding = _get_leading_padding(right_line)
          if leading_padding > 0:
            right_inset = min(leading_padding, inset - left_inset)
            right_line = right_line[right_inset:]
      padding = abs(inset) if inset < 0 else 0
      line = ''.join([left_line, padding_char * padding, right_line])
      lines.append(line)
    left_height = len(left)
    right_height = len(right)
    if left_height > right_height:
      lines.extend(left[right_height:])
    elif right_height > left_height:
      lines.extend(right[left_height:])
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    shortest_line = min(lines, key=len)
    shortest_width = len(shortest_line)
    if shortest_width != longest_width:
      if left_height > right_height:
        lines = _pad_right(lines, longest_width)
      else:
        lines = _pad_left(lines, longest_width)
    return lines

  def _fit_with_padding(
      left, right, preferred_padding, minimum_content_padding=4
  ):
    r"""Concatenates the lines of two `list`s of strings.

    Concatenates the lines of two `list`s of strings by appending each line
    together using a padding. The same padding is used to append each line and
    the padding is calculated starting from the `preferred_padding` without
    going below `minimum_content_padding` on any of the lines. If the two
    `list`s of strings have different lengths, padding will be applied to
    maintain the length of each string in the resulting `list` of strings.

    >>>_fit_with_padding(['aa', 'bb'], ['ccc'])
    ['aa    cccc', 'bb        ']

    >>>_fit_with_padding(['aa          ', 'bb          '], ['          ccc'])
    ['aa    cccc', 'bb        ']

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      A `list` of lines.
    """
    inset = _calculate_inset_from_padding(
        left, right, preferred_padding, minimum_content_padding
    )
    return _fit_with_inset(left, right, inset)

  def _get_node_label(comp):
    """Returns a string for node in the structure of the given `comp`."""
    if isinstance(comp, Block):
      return 'Block'
    elif isinstance(comp, Call):
      return 'Call'
    elif isinstance(comp, CompiledComputation):
      return 'Compiled({})'.format(comp.name)
    elif isinstance(comp, Data):
      return f'Data({id(comp.content)})'
    elif isinstance(comp, Intrinsic):
      return comp.uri
    elif isinstance(comp, Lambda):
      return 'Lambda({})'.format(comp.parameter_name)
    elif isinstance(comp, Reference):
      return 'Ref({})'.format(comp.name)
    elif isinstance(comp, Placement):
      return 'Placement'
    elif isinstance(comp, Selection):
      key = comp.name if comp.name is not None else comp.index
      return 'Sel({})'.format(key)
    elif isinstance(comp, Struct):
      return 'Struct'
    elif isinstance(comp, Literal):
      return f'Lit({comp.value})'
    else:
      raise TypeError('Unexpected type found: {}.'.format(type(comp)))

  def _lines_for_named_comps(named_comps):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named computations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
    """
    lines = ['[']
    for index, (name, comp) in enumerate(named_comps):
      comp_lines = _lines_for_comp(comp)
      if name is not None:
        label = '{}='.format(name)
        comp_lines = _fit_with_padding([label], comp_lines, 0, 0)
      if index == 0:
        lines = _fit_with_padding(lines, comp_lines, 0, 0)
      else:
        lines = _fit_with_padding(lines, [','], 0, 0)
        lines = _fit_with_padding(lines, comp_lines, 1)
    lines = _fit_with_padding(lines, [']'], 0, 0)
    return lines

  def _lines_for_comp(comp):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
    """
    node_label = _get_node_label(comp)

    if isinstance(
        comp,
        (
            CompiledComputation,
            Data,
            Intrinsic,
            Placement,
            Reference,
            Literal,
        ),
    ):
      return [node_label]
    elif isinstance(comp, Block):
      variables_lines = _lines_for_named_comps(comp.locals)
      variables_width = len(variables_lines[0])
      variables_trailing_padding = _get_trailing_padding(variables_lines[0])
      leading_padding = variables_width - variables_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      variables_lines = _concatenate(
          [edge_line], variables_lines, Alignment.LEFT
      )

      result_lines = _lines_for_comp(comp.result)
      result_width = len(result_lines[0])
      leading_padding = _get_leading_padding(result_lines[0]) - 1
      trailing_padding = result_width - leading_padding - 1
      edge_line = '\\{}'.format(padding_char * trailing_padding)
      result_lines = _concatenate([edge_line], result_lines, Alignment.RIGHT)

      preferred_padding = len(node_label)
      lines = _fit_with_padding(
          variables_lines, result_lines, preferred_padding
      )
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif isinstance(comp, Call):
      function_lines = _lines_for_comp(comp.function)
      function_width = len(function_lines[0])
      function_trailing_padding = _get_trailing_padding(function_lines[0])
      leading_padding = function_width - function_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      function_lines = _concatenate([edge_line], function_lines, Alignment.LEFT)

      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument)
        argument_width = len(argument_lines[0])
        leading_padding = _get_leading_padding(argument_lines[0]) - 1
        trailing_padding = argument_width - leading_padding - 1
        edge_line = '\\{}'.format(padding_char * trailing_padding)
        argument_lines = _concatenate(
            [edge_line], argument_lines, Alignment.RIGHT
        )

        preferred_padding = len(node_label)
        lines = _fit_with_padding(
            function_lines, argument_lines, preferred_padding
        )
      else:
        lines = function_lines
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif isinstance(comp, Lambda):
      result_lines = _lines_for_comp(comp.result)
      leading_padding = _get_leading_padding(result_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], result_lines, Alignment.LEFT)
    elif isinstance(comp, Selection):
      source_lines = _lines_for_comp(comp.source)
      leading_padding = _get_leading_padding(source_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], source_lines, Alignment.LEFT)
    elif isinstance(comp, Struct):
      elements = comp.items()
      elements_lines = _lines_for_named_comps(elements)
      leading_padding = _get_leading_padding(elements_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate(
          [node_line, edge_line], elements_lines, Alignment.LEFT
      )
    else:
      raise NotImplementedError(
          f'Unexpected building block found: {type(comp)}.'
      )

  lines = _lines_for_comp(comp)
  lines = [line.rstrip() for line in lines]
  return '\n'.join(lines)
