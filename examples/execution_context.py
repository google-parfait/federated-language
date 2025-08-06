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
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example execution context used to execute a `ConcreteComputation`.

This example demonstrates how to implement an execution context that can execute
a `ConcreteComputation`:

* Federated intrinsics and compositional constructs.
* Logic from other libraries (e.g., TensorFlow, JAX, PyTorch).

IMPORTANT: To highlight the abstract concepts and simplify the implementation,
this example uses Python to represent logic and `cloudpickle` to serialize
Python. However, using Python in this way is an implementation detail and
serializing Python functions may not be suitable for production systems.
"""

from collections.abc import Callable, Mapping, Sequence
import functools
import typing
from typing import NoReturn, Optional, Protocol

import cloudpickle
import federated_language
from federated_language.proto import array_pb2


@typing.runtime_checkable
class _NamedTuple(Protocol):
  """An ABC used to determine if an object is an instance of a `NamedTuple`."""

  @property
  def _fields(self) -> tuple[str, ...]:
    ...

  def _asdict(self) -> dict[str, object]:
    ...


class _ExecutionScope:
  """A data structure that tracks bound symbols and cardinalities for a scope.

  This data structure describes the current scope which may be nested in other
  scopes; the methods of this object will search the current scope and then each
  parent scope until the desired result is found.
  """

  def __init__(
      self,
      scope: '_ExecutionScope' = None,
      symbol_bindings: Optional[Mapping[str, object]] = None,
      cardinalities: Optional[
          Mapping[federated_language.framework.PlacementLiteral, int]
      ] = None,
  ):
    self._parent_scope = scope
    self._symbol_bindings = symbol_bindings
    self._cardinalities = cardinalities

  def resolve_reference(self, name: str) -> object:
    """Returns the object bound to `name`.

    Args:
      name: The name of the object.

    Raises:
      ValueError: If there is no object bound to `name`.
    """
    if self._symbol_bindings is not None and name in self._symbol_bindings:
      return self._symbol_bindings[name]
    elif self._parent_scope is not None:
      return self._parent_scope.resolve_reference(name)
    else:
      raise ValueError(f'Expected a symbol for name: {name}.')

  def get_cardinality(
      self, placement: federated_language.framework.PlacementLiteral
  ) -> int:
    """Returns the cardinality for `placement`.

    Args:
      placement: The placement of the cardinality.

    Raises:
      ValueError: If there is no cardinality for `placement`.
    """
    if self._cardinalities is not None and placement in self._cardinalities:
      return self._cardinalities[placement]
    elif self._parent_scope is not None:
      return self._parent_scope.get_cardinality(placement)
    else:
      raise ValueError(f'Expected a cardinality for placement: {placement}.')


class ExecutionContext(federated_language.framework.AsyncContext):
  """An execution context used to execute a `ConcreteComputation`.

  This execution context prioritizes simplicity and readability over
  performance, and:

  * Executes a `ConcreteComputation` locally in a single Python process.
  * Supports a `ConcreteComputation` created using the `python_computation`
    decorator.
  * Supports all federated compositional constructs.
  * Supports most federated intrinsics.
  * Does not support TensorFlow or XLA constructs.
  * Assumes `computation_pb2.Data` constructs are Python fucntions serialized
    using the `python_computation` decorator.

  For example:

  >>> @python_computation.python_computation([np.int32, np.int32], np.int32)
  >>> def fn(x, y):
  >>>   return x + y
  >>> context = execution_context.ExecutionContext()
  >>> await context.invoke(fn, [1, 2])
  3

  >>> @federated_language.federated_computation(
  >>>     federated_language.FederatedType(
  >>>         np.float32, federated_language.CLIENTS
  >>>     )
  >>> )
  >>> def fn(x):
  >>>   return federated_language.federated_mean(x)
  >>> context = execution_context.ExecutionContext()
  >>> await context.invoke(fn, [1.0, 2.0, 3.0])
  2.0
  """

  def __init__(
      self,
      compiler: Optional[
          Callable[
              [federated_language.framework.ConcreteComputation],
              federated_language.framework.ConcreteComputation,
          ]
      ] = None,
  ):
    self._compiler = compiler

  async def invoke(
      self,
      comp: federated_language.framework.ConcreteComputation,
      arg: Optional[object],
  ) -> object:
    """Invokes the `comp` with the optional `arg`."""
    comp = self._compile(comp)
    comp_pb = comp.to_proto()
    building_block = (
        federated_language.framework.ComputationBuildingBlock.from_proto(
            comp_pb
        )
    )
    cardinalities = {federated_language.SERVER: 1}
    cardinalities |= federated_language.framework.infer_cardinalities(
        arg, comp.type_signature.parameter
    )
    scope = _ExecutionScope(cardinalities=cardinalities)
    fn = self._compute(building_block, scope)

    if not isinstance(fn, Callable):
      raise ValueError(f'Expected `fn` to be `Callable`, found {type(fn)}.')
    result = fn(arg)
    result_type = comp.type_signature.result
    return federated_language.framework.to_structure_with_type(
        result, result_type
    )

  @functools.lru_cache()
  def _compile(
      self,
      comp: federated_language.framework.ConcreteComputation,
  ) -> federated_language.framework.ConcreteComputation:
    if self._compiler is not None:
      comp = self._compiler(comp)
    return comp

  def _compute(
      self,
      buildling_block: federated_language.framework.ComputationBuildingBlock,
      scope: _ExecutionScope,
  ) -> object:
    if isinstance(buildling_block, federated_language.framework.Block):
      return self._compute_block(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Call):
      return self._compute_call(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Data):
      return self._compute_data(buildling_block, scope)
    elif isinstance(
        buildling_block, federated_language.framework.CompiledComputation
    ):
      return self._compute_compiled_computation(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Intrinsic):
      return self._compute_intrinsic(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Lambda):
      return self._compute_lambda(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Literal):
      return self._compute_literal(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Reference):
      return self._compute_reference(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Selection):
      return self._compute_selection(buildling_block, scope)
    elif isinstance(buildling_block, federated_language.framework.Struct):
      return self._compute_struct(buildling_block, scope)
    else:
      raise NotImplementedError(
          f'Unexpected building block found: {type(buildling_block)}.'
      )

  def _compute_block(
      self,
      buildling_block: federated_language.framework.Block,
      scope: _ExecutionScope,
  ) -> object:
    for name, value in buildling_block.locals:
      value = self._compute(value, scope)
      scope = _ExecutionScope(scope, {name: value})
    return self._compute(buildling_block.result, scope)

  def _compute_call(
      self,
      buildling_block: federated_language.framework.Call,
      scope: _ExecutionScope,
  ) -> object:
    fn = self._compute(buildling_block.function, scope)
    if not isinstance(fn, Callable):
      raise ValueError(f'Expected `fn` to be `Callable`, found {type(fn)}.')
    if buildling_block.argument is not None:
      arg = self._compute(buildling_block.argument, scope)
    else:
      arg = None
    return fn(arg)

  def _compute_compiled_computation(
      self,
      buildling_block: federated_language.framework.CompiledComputation,
      scope: _ExecutionScope,
  ) -> NoReturn:
    del scope  # Unused.

    comp_pb = buildling_block.to_proto()
    computation_oneof = comp_pb.WhichOneof('computation')
    if computation_oneof == 'tensorflow':
      raise ValueError('Executing a TensorFlow computation is not supported.')
    elif computation_oneof == 'xla':
      raise ValueError('Executing an XLA computation is not supported.')
    else:
      raise NotImplementedError(
          f'Unexpected computation found: {computation_oneof}.'
      )

  def _compute_data(
      self,
      buildling_block: federated_language.framework.Data,
      scope: _ExecutionScope,
  ) -> Callable[[object], object]:
    del scope  # Unused.

    if not isinstance(
        buildling_block.type_signature, federated_language.FunctionType
    ):
      raise ValueError(
          'Expected `buildling_block` to have a functional type signature,'
          f' found {buildling_block.type_signature}.'
      )

    array_pb = array_pb2.Array()
    buildling_block.content.Unpack(array_pb)
    serialized_fn = federated_language.array_from_proto(array_pb)
    fn = cloudpickle.loads(serialized_fn)

    if buildling_block.type_signature.parameter is None:
      return lambda _: fn()
    else:
      return fn

  def _compute_intrinsic(
      self,
      buildling_block: federated_language.framework.Intrinsic,
      scope: _ExecutionScope,
  ):
    uri = buildling_block.uri
    if uri == federated_language.framework.FEDERATED_AGGREGATE.uri:
      return lambda x: self._federated_aggregate(x, scope)
    elif uri == federated_language.framework.FEDERATED_APPLY.uri:
      return lambda x: self._federated_apply(x, scope)
    elif uri == federated_language.framework.FEDERATED_BROADCAST.uri:
      return lambda x: self._federated_broadcast(x, scope)
    elif uri == federated_language.framework.FEDERATED_EVAL_AT_CLIENTS.uri:
      return lambda x: self._federated_eval(
          x, scope, federated_language.CLIENTS
      )
    elif uri == federated_language.framework.FEDERATED_EVAL_AT_SERVER.uri:
      return lambda x: self._federated_eval(x, scope, federated_language.SERVER)
    elif uri == federated_language.framework.FEDERATED_MAP.uri:
      return lambda x: self._federated_map(x, scope)
    elif uri == federated_language.framework.FEDERATED_MAP_ALL_EQUAL.uri:
      return lambda x: self._federated_map_all_equal(x, scope)
    elif uri == federated_language.framework.FEDERATED_MAX.uri:
      return lambda x: self._federated_max(x, scope)
    elif uri == federated_language.framework.FEDERATED_MEAN.uri:
      return lambda x: self._federated_mean(x, scope)
    elif uri == federated_language.framework.FEDERATED_MIN.uri:
      return lambda x: self._federated_min(x, scope)
    elif uri == federated_language.framework.FEDERATED_SUM.uri:
      return lambda x: self._federated_sum(x, scope)
    elif uri == federated_language.framework.FEDERATED_VALUE_AT_CLIENTS.uri:
      return lambda x: self._federated_value(x, scope)
    elif uri == federated_language.framework.FEDERATED_VALUE_AT_SERVER.uri:
      return lambda x: self._federated_value(x, scope)
    elif uri == federated_language.framework.FEDERATED_ZIP_AT_CLIENTS.uri:
      return lambda x: self._federated_zip_clients(x, scope)
    elif uri == federated_language.framework.FEDERATED_ZIP_AT_SERVER.uri:
      return lambda x: self._federated_zip_server(x, scope)
    else:
      raise NotImplementedError(f'Unexpected intrinsic found: {uri}.')

  def _compute_lambda(
      self,
      buildling_block: federated_language.framework.Lambda,
      scope: _ExecutionScope,
  ) -> Callable[[object], object]:

    def _fn(arg: object) -> object:
      nonlocal scope
      if buildling_block.parameter_type is not None:
        scope = _ExecutionScope(scope, {buildling_block.parameter_name: arg})
      return self._compute(buildling_block.result, scope)

    return _fn

  def _compute_literal(
      self,
      buildling_block: federated_language.framework.Literal,
      scope: _ExecutionScope,
  ) -> object:
    del scope  # Unused.

    return buildling_block.value

  def _compute_reference(
      self,
      buildling_block: federated_language.framework.Reference,
      scope: _ExecutionScope,
  ) -> object:
    return scope.resolve_reference(buildling_block.name)

  def _compute_selection(
      self,
      buildling_block: federated_language.framework.Selection,
      scope: _ExecutionScope,
  ) -> object:
    index = buildling_block.as_index()
    source = self._compute(buildling_block.source, scope)

    if isinstance(source, _NamedTuple):
      name = source._fields[index]
      return getattr(source, name)
    elif isinstance(source, Mapping):
      key = list(source)[index]
      return source[key]
    elif isinstance(source, Sequence):
      return source[index]
    else:
      raise NotImplementedError(f'Unexpected source found: {type(source)}.')

  def _compute_struct(
      self,
      buildling_block: federated_language.framework.Struct,
      scope: _ExecutionScope,
  ) -> list[object]:
    return [self._compute(x, scope) for x in buildling_block]

  def _federated_aggregate(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    values, zero, accumulate, _, report = arg
    total = zero
    for value in values:
      total = accumulate([total, value])
    return report(total)

  def _federated_apply(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    fn, arg = arg
    return fn(arg)

  def _federated_broadcast(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return arg

  def _federated_eval(
      self,
      arg,
      scope: _ExecutionScope,
      placement: federated_language.framework.PlacementLiteral,
  ):
    fn = arg
    cardinality = scope.get_cardinality(placement)
    if cardinality == 1:
      return fn(None)
    else:
      return [fn(None) for _ in range(cardinality)]

  def _federated_map(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    fn, arg = arg
    return [fn(x) for x in arg]

  def _federated_map_all_equal(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    fn, arg = arg
    return fn(arg)

  def _federated_max(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return max(arg)

  def _federated_mean(self, arg, scope: _ExecutionScope):
    total = self._federated_sum(arg, scope)
    return total / len(arg)

  def _federated_min(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return min(arg)

  def _federated_sum(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return sum(arg)

  def _federated_value(
      self,
      arg,
      scope: _ExecutionScope,
  ):
    del scope  # Unused.

    return arg

  def _federated_zip_clients(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return list(zip(*arg))

  def _federated_zip_server(self, arg, scope: _ExecutionScope):
    del scope  # Unused.

    return arg
