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
"""An example decorator that creates a `Computation` from a Python function.

This example demonstrates how to express logic from other libraries (e.g.,
TensorFlow, JAX, PyTorch) as a Federated Language `Computation`.

WARNING: This example uses `cloudpickle` to serialize Python. This purpose of
this example is to highlight concepts of the Federated Language, serializing
Python using `cloudpickle` is not recommended for all systems and should thought
of as an implementation detail.
"""

from collections.abc import Callable, Mapping, Sequence
import functools
import inspect
from typing import Optional, TypeVar, Union

import cloudpickle
import federated_language
from federated_language.proto import computation_pb2

from google.protobuf import any_pb2


T = TypeVar('T')
Structure = Union[
    Sequence[T],
    Mapping[str, T],
]


def _create_computation_from_fn(
    fn: Callable[..., object],
    parameter_type: federated_language.Type,
    result_type: federated_language.Type,
) -> computation_pb2.Computation:
  """Returns a `computation_pb2.Computation` for the Python function.

  Args:
    fn: A Python function to use.
    parameter_type: The parameter type of the Python function.
    result_type: The result type of the  Python function.
  """
  fn = federated_language.framework.wrap_as_zero_or_one_arg_callable(
      fn, parameter_type
  )

  type_spec = federated_language.FunctionType(parameter_type, result_type)
  serialized_fn = cloudpickle.dumps(fn)
  array_pb = federated_language.array_to_proto(serialized_fn)
  content_pb = any_pb2.Any()
  content_pb.Pack(array_pb)
  data_pb = computation_pb2.Data(content=content_pb)
  return computation_pb2.Computation(
      type=type_spec.to_proto(),
      data=data_pb,
  )


def _create_concrete_computation_from_fn(
    fn: Callable[..., object],
    parameter_type: Optional[federated_language.Type],
    result_type: federated_language.Type,
) -> federated_language.framework.ConcreteComputation:
  """Returns a `ConcreteComputation` for the Python function.

  Args:
    fn: A Python function to use.
    parameter_type: The parameter type of the Python function.
    result_type: The result type of the  Python function.
  """
  comp_pb = _create_computation_from_fn(fn, parameter_type, result_type)
  context_stack = federated_language.framework.get_context_stack()
  annotated_type = federated_language.FunctionType(parameter_type, result_type)
  return federated_language.framework.ConcreteComputation(
      computation_proto=comp_pb,
      context_stack=context_stack,
      annotated_type=annotated_type,
  )


def _check_parameter_types(
    parameter_types: Optional[Structure[object]],
    parameters: Mapping[str, inspect.Parameter],
) -> None:
  """Checks that the `parameter_types` match the `parameters` of the function.

  Args:
    parameter_types: The parameter types to check.
    parameters: The parameters of the Python function.

  Raises:
    ValueError: If the parameter types do not match the `parameters` of the
      function.
  """
  if (parameter_types is None and parameters) or (
      parameter_types is not None and len(parameter_types) != len(parameters)
  ):
    if parameter_types is None:
      parameter_types_length = 0
    else:
      parameter_types_length = len(parameter_types)
    raise ValueError(
        f'Expected the number of `parameter_types` {parameter_types_length} to'
        f' match the number of `parameters` {len(parameters)}.'
    )


def python_computation(
    parameter_types: Optional[Structure[object]],
    result_type: object,
) -> Callable[[Callable[..., object]], federated_language.Computation]:
  """A decorator factory that creates a `Computation` from a Python function.

  It is infeasible to infer the `return_type` of a Python function because of
  the dynamic nature of the language. It is easier to infer the return type of
  functions from other libraries (e.g., TensorFlow, JAX, PyTorch) because those
  libraries limit the expressiveness of Python.

  To be consistent with how the `return_type` is handled, this decorator
  intentionally chooses not to infer the `parameter_types` of the Python
  function. Additionally, to make the usage easier to reason about, if
  `parameter_types` is not None, then it must be a collection with a length
  matching the number of parameters defined by the Python function.

  For example:

  >>> @python_computation.python_computation([np.int32], np.int32)
  >>> def fn(x):
  >>>   return x + 1

  >>> @python_computation.python_computation([np.int32, np.int32], np.int32)
  >>> def fn(x, y):
  >>>   return x + y

  Args:
    parameter_types: The parameter types of the Python function. Can be an
      collection of any type that is convertable to a `federated_language.Type`
      or `None` if the Python function has no parameters. An empty collection is
      treated as `None`.
    result_type: The result type of the Python function; can be any type that is
      convertable to a `federated_language.Type`.

  Returns:
    A decorator that creates a `Computation` from a Python function.

  Raises:
    ValueError: If the `parameter_types` do not match the `parameters` of the
      function.
  """

  def _decorator(fn: Callable[..., object]) -> federated_language.Computation:
    nonlocal parameter_types
    nonlocal result_type

    parameters = inspect.signature(fn).parameters
    _check_parameter_types(parameter_types, parameters)
    if not parameter_types:
      parameter_type = None
    else:
      if len(parameter_types) == 1:
        parameter_types, *_ = parameter_types
      parameter_type = federated_language.to_type(parameter_types)
    result_type = federated_language.to_type(result_type)

    comp = _create_concrete_computation_from_fn(fn, parameter_type, result_type)
    return functools.update_wrapper(comp, fn, updated=())  # pytype: disable=bad-return-type

  return _decorator
