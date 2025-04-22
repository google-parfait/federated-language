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
"""Definition of a federated computation."""

from typing import Optional

from federated_language.computation import computation_impl
from federated_language.computation import computation_wrapper
from federated_language.computation import function_utils
from federated_language.context_stack import context_stack_impl
from federated_language.federated_context import federated_computation_utils


def _federated_computation_wrapper_fn(
    fn,
    parameter_type,
    unpack: Optional[bool],
    name: Optional[str] = None,
    **kwargs
):
  """Wrapper function to plug orchestration logic into the framework."""
  del kwargs  # Unused.
  if parameter_type is None:
    parameter_name = None
  else:
    parameter_name = 'arg'
  fn = function_utils.wrap_as_zero_or_one_arg_callable(
      fn, parameter_type, unpack
  )
  context_stack = context_stack_impl.context_stack
  target_lambda, extra_type_spec = (
      federated_computation_utils.zero_or_one_arg_fn_to_building_block(
          fn,
          parameter_name,
          parameter_type,
          context_stack,
          suggested_name=name,
      )
  )
  return computation_impl.ConcreteComputation(
      computation_proto=target_lambda.to_proto(),
      context_stack=context_stack,
      annotated_type=extra_type_spec,
  )


federated_computation = computation_wrapper.ComputationWrapper(
    _federated_computation_wrapper_fn
)
federated_computation.__doc__ = """Decorates/wraps Python functions as federated/composite computations.

  The term *federated computation* as used here refers to any computation that
  uses API programming abstractions. Examples of such computations may include
  federated training or federated evaluation that involve both client-side and
  server-side logic and involve network communication. However, this
  decorator/wrapper can also be used to construct composite computations that
  only involve local processing on a client or on a server.

  The main feature that distinguishes *federated computation* function bodies
  in Python from the bodies of TensorFlow defuns is that whereas in the latter,
  one slices and dices `tf.Tensor` instances using a variety of TensorFlow ops,
  in the former one slices and dices `federated_language.Value` instances using operators.

  The supported modes of usage are identical to those for
  `federated_language.tensorflow.computation`.

  Example:

    ```python
    @federated_language.federated_computation((federated_language.FunctionType(np.int32, np.int32), np.int32))
    def foo(f, x):
      return f(f(x))
    ```

    The above defines `foo` as a function that takes a tuple consisting of an
    unary integer operator as the first element, and an integer as the second
    element, and returns the result of applying the unary operator to the
    integer twice. The body of `foo` does not contain federated communication
    operators, but we define it with `federated_language.federated_computation` as it can be
    used as building block in any section of code (except inside sections
    of pure TensorFlow logic).

  Args:
    *args: Either a Python function, or Federated Language type spec, or both
      (function first), or neither. See also
      `federated_language.tensorflow.computation` for an extended documentation.

  Returns:
    If invoked with a function as an argument, returns an instance of a
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter. See
    also `federated_language.tensorflow.computation` for an extended documentation.
  """
