# Copyright 2025 Google LLC
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

from typing import NamedTuple
import unittest
from unittest import mock

from absl.testing import absltest
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import numpy as np

import execution_context
import python_computation


def _assert_computation_contains(
    comp: federated_language.framework.ConcreteComputation,
    classinfo: type[federated_language.framework.ComputationBuildingBlock],
) -> None:
  predicate = lambda x: isinstance(x, classinfo)
  if not federated_language.framework.computation_contains(comp, predicate):
    raise AssertionError(
        f'Expected `comp` to contain {classinfo}, found {comp.to_proto()}.'
    )


class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  async def test_compiler_called_once_for_comp(self):

    def _compiler(comp):
      return comp

    mock_compiler = mock.MagicMock(spec_set=_compiler, wraps=_compiler)
    context = execution_context.ExecutionContext(mock_compiler)

    @python_computation.python_computation([np.int32], np.int32)
    def _fn(x):
      return x

    await context.invoke(_fn, 1)
    await context.invoke(_fn, 2)
    mock_compiler.assert_called_once_with(_fn)


class ExecutionContextPythonComputationTest(unittest.IsolatedAsyncioTestCase):

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_none(self):

    @python_computation.python_computation(None, np.int32)
    def _fn():
      return 1

    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_empty(self):

    @python_computation.python_computation([], np.int32)
    def _fn():
      return 1

    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_one_int(self):

    @python_computation.python_computation([np.int32], np.int32)
    def _fn(x):
      return x + 1

    result = await _fn(1)
    self.assertEqual(result, 2)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_one_list(self):

    @python_computation.python_computation(
        [[np.int32, np.float32]], [np.int32, np.float32]
    )
    def _fn(x):
      return [x[0] + 1, x[1] + 1.0]

    result = await _fn([1, 2.0])
    self.assertEqual(result, [2, 3.0])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_one_dict(self):

    @python_computation.python_computation(
        [[('a', np.int32), ('b', np.float32)]],
        [('a', np.int32), ('b', np.float32)],
    )
    def _fn(x):
      return {
          'a': x['a'] + 1,
          'b': x['b'] + 1.0,
      }

    result = await _fn({'a': 1, 'b': 2.0})
    self.assertEqual(result, {'a': 2, 'b': 3.0})

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_one_named_tuple(
      self,
  ):

    class _Foo(NamedTuple):
      a: int
      b: float

    @python_computation.python_computation([_Foo], _Foo)
    def _fn(x):
      return _Foo(x.a + 1, x.b + 1.0)

    result = await _fn(_Foo(1, 2.0))
    self.assertEqual(result, _Foo(2, 3.0))

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_multiple_unnamed(self):

    @python_computation.python_computation(
        [np.int32, np.float32], [np.int32, np.float32]
    )
    def _fn(x, y):
      return [x + 1, y + 1.0]

    result = await _fn(1, 2.0)
    self.assertEqual(result, [2, 3.0])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_with_args_multiple_named(self):

    @python_computation.python_computation(
        {'x': np.int32, 'y': np.float32},
        [np.int32, np.float32],
    )
    def _fn(x, y):
      return [x + 1, y + 1.0]

    result = await _fn(1, 2.0)
    self.assertEqual(result, [2, 3.0])


class ExecutionContextFederatedConstructsTest(unittest.IsolatedAsyncioTestCase):

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_block(self):

    @python_computation.python_computation(None, np.int32)
    def _value():
      return 1

    @federated_language.federated_computation()
    def _fn():
      return _value()

    _assert_computation_contains(_fn, federated_language.framework.Block)
    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_call_arg(self):

    @python_computation.python_computation(None, np.int32)
    def _value():
      return 1

    @federated_language.federated_computation()
    def _fn():
      return _value()

    _assert_computation_contains(_fn, federated_language.framework.Call)
    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_call_no_arg(self):

    @python_computation.python_computation(None, np.int32)
    def _value():
      return 1

    @federated_language.federated_computation()
    def _fn():
      return _value()

    _assert_computation_contains(_fn, federated_language.framework.Call)
    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_lambda(self):

    @federated_language.federated_computation(np.int32)
    def _fn(x):
      return x

    _assert_computation_contains(_fn, federated_language.framework.Lambda)
    result = await _fn(1)
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_literal(self):

    @federated_language.federated_computation()
    def _fn():
      return 1

    _assert_computation_contains(_fn, federated_language.framework.Literal)
    result = await _fn()
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_reference(self):

    @federated_language.federated_computation(np.int32)
    def _fn(x):
      return x

    _assert_computation_contains(_fn, federated_language.framework.Reference)
    result = await _fn(1)
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_selection_list(self):

    @federated_language.federated_computation([np.int32, np.float32])
    def _fn(x):
      return x[0]

    _assert_computation_contains(_fn, federated_language.framework.Selection)
    result = await _fn([1, 2.0])
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_selection_dict(self):

    @federated_language.federated_computation(
        [('a', np.int32), ('b', np.float32)]
    )
    def _fn(x):
      return x['a']

    _assert_computation_contains(_fn, federated_language.framework.Selection)
    result = await _fn({'a': 1, 'b': 2.0})
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_selection_named_tuple(self):

    class _Foo(NamedTuple):
      a: int
      b: float

    @federated_language.federated_computation(_Foo)
    def _fn(x):
      return x['a']

    _assert_computation_contains(_fn, federated_language.framework.Selection)
    result = await _fn(_Foo(1, 2.0))
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_struct(self):

    @federated_language.federated_computation(np.int32)
    def _fn(x):
      return [x, x]

    _assert_computation_contains(_fn, federated_language.framework.Struct)
    result = await _fn(1)
    self.assertEqual(result, [1, 1])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_raises_value_error_with_data(self):
    tensor_type_pb = computation_pb2.TensorType(dtype=data_type_pb2.DT_INT32)
    result_type_pb = computation_pb2.Type(tensor=tensor_type_pb)
    data_pb = computation_pb2.Data()
    result_pb = computation_pb2.Computation(
        type=result_type_pb,
        data=data_pb,
    )
    function_type_pb = computation_pb2.FunctionType(result=result_type_pb)
    type_pb = computation_pb2.Type(function=function_type_pb)
    fn_pb = computation_pb2.Lambda(
        parameter_name=None,
        result=result_pb,
    )
    comp_pb = computation_pb2.Computation(
        type=type_pb,
        **{'lambda': fn_pb},
    )
    context_stack = federated_language.framework.get_context_stack()
    fn = federated_language.framework.ConcreteComputation(
        computation_proto=comp_pb,
        context_stack=context_stack,
    )

    with self.assertRaises(ValueError):
      await fn()

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_raises_value_error_tensorflow(self):
    tensor_type_pb = computation_pb2.TensorType(dtype=data_type_pb2.DT_INT32)
    result_type_pb = computation_pb2.Type(tensor=tensor_type_pb)
    function_type_pb = computation_pb2.FunctionType(result=result_type_pb)
    type_pb = computation_pb2.Type(function=function_type_pb)
    tensorflow_pb = computation_pb2.TensorFlow()
    comp_pb = computation_pb2.Computation(
        type=type_pb,
        tensorflow=tensorflow_pb,
    )
    context_stack = federated_language.framework.get_context_stack()
    fn = federated_language.framework.ConcreteComputation(
        computation_proto=comp_pb,
        context_stack=context_stack,
    )

    with self.assertRaises(ValueError):
      await fn()

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_raises_value_error_xla(self):
    tensor_type_pb = computation_pb2.TensorType(dtype=data_type_pb2.DT_INT32)
    result_type_pb = computation_pb2.Type(tensor=tensor_type_pb)
    function_type_pb = computation_pb2.FunctionType(result=result_type_pb)
    type_pb = computation_pb2.Type(function=function_type_pb)
    xla_pb = computation_pb2.Xla()
    comp_pb = computation_pb2.Computation(
        type=type_pb,
        xla=xla_pb,
    )
    context_stack = federated_language.framework.get_context_stack()
    fn = federated_language.framework.ConcreteComputation(
        computation_proto=comp_pb,
        context_stack=context_stack,
    )

    with self.assertRaises(ValueError):
      await fn()


class ExecutionContextFederatedIntrinsicsTest(unittest.IsolatedAsyncioTestCase):

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_aggregate(self):

    class _Foo(NamedTuple):
      sum: float
      count: int

    @python_computation.python_computation(None, _Foo)
    def _zero() -> _Foo:
      return _Foo(sum=0.0, count=0)

    @python_computation.python_computation([_Foo, np.float32], _Foo)
    def _accumulate(a: _Foo, value: float) -> _Foo:
      return _Foo(sum=a.sum + value, count=a.count + 1)

    @python_computation.python_computation([_Foo, _Foo], _Foo)
    def _merge(a: _Foo, b: _Foo) -> _Foo:
      return _Foo(sum=a.sum + b.sum, count=a.count + b.count)

    @python_computation.python_computation([_Foo], np.float32)
    def _report(a: _Foo) -> float:
      return a.sum / a.count

    @federated_language.federated_computation(
        federated_language.FederatedType(np.float32, federated_language.CLIENTS)
    )
    def _fn(x):
      zero = _zero()
      return federated_language.federated_aggregate(
          x, zero, _accumulate, _merge, _report
      )

    result = await _fn([1.0, 2.0, 3.0])
    self.assertEqual(result, 2.0)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_broadcast(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.SERVER)
    )
    def _fn(x):
      return federated_language.federated_broadcast(x)

    result = await _fn(1)
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_eval_at_clients(self):

    @python_computation.python_computation(None, [np.int32, np.int32, np.int32])
    def _value():
      return [1, 2, 3]

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def _fn(x):
      del x  # Unused.
      return federated_language.federated_eval(
          _value, federated_language.CLIENTS
      )

    # NOTE: The x is used by the context to infer the cardinality of values
    # placed at `federated_language.CLIENTS`.
    result = await _fn([None, None, None])
    self.assertEqual(result, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_eval_at_server(self):

    @python_computation.python_computation(None, [np.int32, np.int32, np.int32])
    def _value():
      return [1, 2, 3]

    @federated_language.federated_computation()
    def _fn():
      return federated_language.federated_eval(
          _value, federated_language.SERVER
      )

    result = await _fn()
    self.assertEqual(result, [1, 2, 3])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_map(self):

    @python_computation.python_computation([np.int32], np.int32)
    def _add_one(x):
      return x + 1

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def _fn(x):
      return federated_language.federated_map(_add_one, x)

    result = await _fn([1, 2, 3])
    self.assertEqual(result, [2, 3, 4])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_max(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def _fn(x):
      return federated_language.federated_max(x)

    result = await _fn([1, 2, 3])
    self.assertEqual(result, 3)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_mean(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.float32, federated_language.CLIENTS)
    )
    def _fn(x):
      return federated_language.federated_mean(x)

    result = await _fn([1.0, 2.0, 3.0])
    self.assertEqual(result, 2.0)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_min(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def _fn(x):
      return federated_language.federated_min(x)

    result = await _fn([1, 2, 3])
    self.assertEqual(result, 1)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_sum(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def _fn(x):
      return federated_language.federated_sum(x)

    result = await _fn([1, 2, 3])
    self.assertEqual(result, 6)

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_value_at_clients(self):

    @federated_language.federated_computation()
    def _fn():
      return federated_language.federated_value(
          [1, 2, 3], federated_language.CLIENTS
      )

    result = await _fn()
    self.assertEqual(result, [1, 2, 3])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_value_at_server(self):

    @federated_language.federated_computation()
    def _fn():
      return federated_language.federated_value(
          [1, 2, 3], federated_language.SERVER
      )

    result = await _fn()
    self.assertEqual(result, [1, 2, 3])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_zip_at_clients(self):

    @federated_language.federated_computation([
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
    ])
    def _fn(x):
      return federated_language.federated_zip(x)

    result = await _fn([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(result, [[1, 4], [2, 5], [3, 6]])

  @federated_language.framework.with_context(execution_context.ExecutionContext)
  async def test_invoke_returns_result_federated_zip_at_server(self):

    @federated_language.federated_computation([
        federated_language.FederatedType(
            [np.int32, np.int32, np.int32], federated_language.SERVER
        ),
        federated_language.FederatedType(
            [np.int32, np.int32, np.int32], federated_language.SERVER
        ),
    ])
    def _fn(x):
      return federated_language.federated_zip(x)

    result = await _fn([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])


if __name__ == '__main__':
  absltest.main()
