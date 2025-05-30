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
"""Utilities for constructing decorators for functions and `tf.function`s."""

import collections
from collections.abc import Callable, Iterable, Mapping
import inspect
from typing import Optional

from federated_language.computation import computation_base
from federated_language.computation import polymorphic_computation
from federated_language.types import computation_types
from federated_language.types import type_conversions


def _parameters(fn) -> Mapping[str, inspect.Parameter]:
  return inspect.signature(fn).parameters


def _check_parameters(parameters: Mapping[str, inspect.Parameter]) -> None:
  """Ensure only non-varargs positional-or-keyword arguments."""
  for parameter in parameters.values():
    if parameter.default is not inspect.Parameter.empty:
      # We don't have a way to build defaults into the function's type.
      raise TypeError(
          'Default parameters are not supported. Found parameter '
          f'`{parameter.name}` with default value {parameter.default}'
      )
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
      # We don't have a way to encode positional-only into the function's type.
      raise TypeError(
          'Positional-only parameters are not supported. Found parameter '
          f'`{parameter.name}` which appears before a `/` entry.'
      )
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
      # We don't have a way to encode keyword-only into the function's type.
      raise TypeError(
          'Keyword-only arguments are not supported. Found parameter '
          f'`{parameter.name}` which appears after a `*` or `*args` entry.'
      )
    if parameter.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
      # For concrete functions, we can't determine at tracing time which
      # arguments should be bundled into args vs. kwargs, since arguments can
      # be passed by position *or* by keyword at later call sites.
      raise TypeError(
          'Parameter varargs are not supported. Found varargs parameter '
          f'`{parameter.name}`.'
      )
    if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
      raise AssertionError(f'Unexpected parameter kind: {parameter.kind}')


def _wrap_polymorphic(
    fn,
    wrapper_fn: Callable[..., computation_base.Computation],
    infer_type_fn,
) -> polymorphic_computation.PolymorphicComputation:
  """Wraps `fn` in `wrapper_fn` at invocation time."""
  try:
    name = fn.__name__
  except AttributeError:
    name = None

  def _polymorphic_wrapper(
      parameter_type: computation_types.Type, unpack: Optional[bool]
  ):
    return wrapper_fn(fn, parameter_type, unpack=unpack, name=name)

  return polymorphic_computation.PolymorphicComputation(
      _polymorphic_wrapper, infer_type_fn
  )


def _wrap_concrete(
    fn,
    wrapper_fn: Callable[..., computation_base.Computation],
    parameter_type,
) -> computation_base.Computation:
  """Wraps `fn` in `wrapper_fn` given the provided `parameter_type`."""
  try:
    name = fn.__name__
  except AttributeError:
    name = None

  concrete_fn = wrapper_fn(fn, parameter_type, unpack=None, name=name)
  result_parameter_type = concrete_fn.type_signature.parameter
  if (
      result_parameter_type is not None
      and not result_parameter_type.is_equivalent_to(parameter_type)
  ):
    raise TypeError(
        'Expected a concrete function that takes parameter {}, got one '
        'that takes {}.'.format(
            str(parameter_type), str(concrete_fn.type_signature.parameter)
        )
    )
  return concrete_fn


def _parameter_type(
    parameters: Mapping[str, inspect.Parameter],
    parameter_types: tuple[computation_types.Type, ...],
) -> Optional[computation_types.Type]:
  """Bundle any user-provided parameter types into a single argument type."""
  parameter_names = parameters.keys()
  if not parameter_types and not parameters:
    return None
  if len(parameter_types) == 1:
    parameter_type = parameter_types[0]
    if parameter_type is None and not parameters:
      return None
    if len(parameters) == 1:
      return parameter_type
    # There is a single parameter type but multiple parameters.
    if not isinstance(parameter_type, computation_types.StructType) or len(
        parameter_type
    ) != len(parameters):
      raise TypeError(
          f'Function with {len(parameters)} parameters must have a parameter '
          'type with the same number of parameters. Found parameter type '
          f'{parameter_type}.'
      )
    name_list_from_types = [
        n for n, _ in parameter_type.items() if n is not None
    ]
    if name_list_from_types:
      if len(name_list_from_types) != len(parameter_type):
        raise TypeError(
            'Types with both named and unnamed fields cannot be unpacked into '
            f'argument lists. Found parameter type {parameter_type}.'
        )
      if set(name_list_from_types) != set(parameter_names):
        raise TypeError(
            'Function argument names must match field names of parameter type. '
            f'Found argument names {parameter_names}, which do not match '
            f'{name_list_from_types}, the top-level fields of the parameter '
            f'type {parameter_type}.'
        )
      # The provided parameter type has all named fields which exactly match
      # the names of the function's parameters.
      return parameter_type
    else:
      # The provided parameter type has no named fields. Apply the names from
      # the function parameters.
      parameter_types = (v for (_, v) in parameter_type.items())
      return computation_types.StructWithPythonType(
          list(zip(parameter_names, parameter_types)), collections.OrderedDict
      )
  elif len(parameters) == 1:
    # If there are multiple provided argument types but the function being
    # decorated only accepts a single argument, tuple the arguments together.
    return computation_types.StructType(parameter_types)
  if len(parameters) != len(parameter_types):
    raise TypeError(
        f'Function with {len(parameters)} parameters is '
        f'incompatible with provided argument types {parameter_types}.'
    )
  # The function has `n` parameters and `n` parameter types.
  # Zip them up into a structure using the names from the function as keys.
  return computation_types.StructWithPythonType(
      list(zip(parameter_names, parameter_types)), collections.OrderedDict
  )


def _wrap(
    fn,
    wrapper_fn: Callable[..., computation_base.Computation],
    parameter_types: tuple[computation_types.Type, ...],
    infer_type_fn: Callable[[object], computation_types.Type],
):
  """Wraps a possibly-polymorphic `fn` in `wrapper_fn`.

  If `parameter_type` is `None` and `fn` takes any arguments (even with default
  values), `fn` is inferred to be polymorphic and won't be passed to
  `wrapper_fn` until invocation time (when concrete parameter types are
  available).

  `wrapper_fn` must accept three positional arguments and one defaulted argument
  `name`:

  * `target_fn`, the Python function to be wrapped.

  * `parameter_types`, the user-provded list of parameter types.

  * `unpack`, an argument which will be passed on to
    `function_utils.wrap_as_zero_or_one_arg_callable` when wrapping `target_fn`.
    See that function for details.

  * Optional `name`, the name of the function that is being wrapped (only for
    debugging purposes).

  Args:
    fn: The function or tf.function to wrap as a computation.
    wrapper_fn: The Python callable that performs actual wrapping. The object to
      be returned by this function should be an instance of a
      `federated_language.framework.ConcreteComputation`.
    parameter_types: Types of any arguments to `fn`.
    infer_type_fn: ...

  Returns:
    Either the result of wrapping (an object that represents the computation),
    or a polymorphic callable that performs wrapping upon invocation based on
    argument types. The returned function still may accept multiple
    arguments (it has not yet had
    `function_uils.wrap_as_zero_or_one_arg_callable` applied to it).
  """
  parameters = _parameters(fn)
  # NOTE: many of the properties checked here are only necessary for
  # non-polymorphic computations whose type signatures must be resolved
  # prior to use. However, we continue to enforce these requirements even
  # in the polymorphic case in order to avoid creating an inconsistency.
  _check_parameters(parameters)

  if parameters and not parameter_types:
    # There is no type specification, and the function/tf.function declares
    # parameters. Create a polymorphic template.
    wrapped_fn = _wrap_polymorphic(fn, wrapper_fn, infer_type_fn)
  else:
    # Either we have a concrete parameter type, or this is no-arg function.
    parameter_type = _parameter_type(parameters, parameter_types)
    wrapped_fn = _wrap_concrete(fn, wrapper_fn, parameter_type)

  # When applying a decorator, the __doc__ attribute with the documentation
  # in triple-quotes is not automatically transferred from the function on
  wrapped_fn.__doc__ = getattr(fn, '__doc__', None)
  return wrapped_fn


def _is_function(obj: object) -> bool:
  # Type specifications are supported (i.e. objects that can be turned into a
  # `federated_language.Type`) as arguments to a computation decorator. In some
  # cases those type specifications (e.g. np.int32) are a `type`, making them
  # `Callable`, but they should not be treated as the function being decorated.
  if isinstance(obj, type):
    return False
  return isinstance(obj, Callable)


class ComputationReturnedNoneError(ValueError):
  """Error for computations which return `None` or do not return."""

  def __init__(self, fn):
    code = fn.__code__
    line_number = code.co_firstlineno
    filename = code.co_filename
    message = (
        f'The function defined on line {line_number} of file {filename} '
        "returned `None` (or didn't explicitly `return` at all), but "
        'computations must return some non-`None` value.'
    )
    super().__init__(message)


class ComputationWrapper:
  """A class for creating wrappers that convert functions into computations.

  This class builds upon the _wrap() function defined above, adding on
  functionality shared between the computation and `federated_computation`
  decorators. The shared functionality includes relating formal Python function
  parameters and call arguments to types, packing and unpacking arguments,
  verifying types, and support for polymorphism.

  Here's how one can use `ComputationWrapper` to construct a decorator/wrapper
  named `xyz`:

  ```python
  def my_wrapper_fn(target_fn, parameter_type, unpack, name=None):
    ...
  xyz = computation_wrapper.ComputationWrapper(my_wrapper_fn)
  ```

  The resulting `xyz` can then be used either as an `@xyz(...)` decorator or as
  a manual wrapping function: `wrapped_fn = xyz(my_func, ...)`. The latter
  method may be preferable when using functions from an external module or
  for wrapping an anonymous lambda.

  The decorator can be used in two ways:
  1. Invoked with a single positional argument specifying the types of the
     function's arguments (`@xyz(some_argument_type)`).
  2. Invoked with no arguments (`@xyz` or `@xyz()`). This is used for functions
     which take no arguments, or functions which are polymorphic (used with
     multiple different argument types).

  Here's how the decorator behaves in each case:

  If the user specifies a tuple type in an unbundled form (simply by listing the
  types of its constituents as separate arguments), the tuple type is formed on
  the user's behalf for convenience.

  1. When the decorator is invoked with positional arguments:

     ```python
     @xyz(('x', np.int32), ('y', np.int32))
     ```

     The decorator arguments must be instances of `types.Type`, or something
     convertible to it by `types.to_type()`. The arguments are interpreted as
     the specification of the parameter of the computation being constructed by
     the decorator. Since the formal parameter to computations is always a
     single argument, multiple arguments to the decorator will be packed into a
     tuple type. This means that the following two invocations behave the same:

     ```
     @xyz(('x', np.int32), ('y', np.int32)) # gets packed into the below
     @xyz((('x', np.int32), ('y', np.int32)))
     ```

     In the above example, the computation will accept as an argument a pair
     of integers named `x` and `y`.

     The function being decorated this way must declare at least one parameter.

     a. If the Python function declares only one parameter, that parameter will
        receive all arguments packed into a single value:

        ```python
        @xyz(('x', np.int32), ('y', np.int32))
        def my_comp(coord):
          ... # use `coord.x` and `coord.y`
        ```

     b. If the Python function declares multiple parameters, the computation's
        parameter type must be convertible to type
        `federated_language.StructType`
        (usually a list containing types or pairs of `(str, types.Type)`.

        ```python
        # With explicitly named parameters
        @xyz(('x', np.int32), ('y', np.int32))
        def my_comp(x, y):
          ... # use `x` and `y`

        # Without explicitly named parameters
        @xyz(np.int32, np.int32)
        def my_comp(x, y):
          ... # use `x` and `y`
        ```

        The number and order of parameters in the decorator arguments and the
        Python function must match. For named elements, the names in the
        decorator and the Python function must also match.

  2. When the decorator is specified without arguments (`@xyz` or `@xyz()`):

     a. If the Python function declares no parameters, the decorator constructs
        a no-parameter computation, as in the following example:

        ```python
        @xyz
        def my_comp():
          ...
        ```

     b. If the function does declare at least one parameter, it is treated as a
        polymorphic function that's instantiated in each concrete context in
        which it's used based on the types of its arguments. The decorator still
        handles the plumbing and parameter type inference.

        For example:

        ```python
        @xyz
        def my_comp(x, y):
          ...
        ```

        In this case, `my_comp` becomes a polymorphic callable, with the actual
        construction postponed. Suppose it's then used as follows, e.g., in an
        orchestration context:

        ```python
        my_comp(5.0, True)
        ```

        At the time of invocation, the decorator uses the information contained
        in the call arguments 5.0 and True to infer the computation's parameter
        type signature, and once the types have been determined, proceeds in
        exactly the same manner as already described in (1) above.

        It is important to note that the arguments of the invocation are not
        simply passed into the body of the Python function being decorated.
        The parameter type inference step is all that differs between the
        polymorphic case and case (1) above.

        Polymorphic functions are the only case where no constraints exist on
        the kinds of arguments that may be present: declaring default values,
        `*args` or `**kwargs`, and any combination of those are valid. The
        mapping is resolved at invocation time based on arguments of the call,
        as in the example below:

        ```python
        @xyz
        def my_comp(x, y=True, *args, **kwargs):
          ...

        my_comp(1, False, 2, 3, 'foo', name='bar')
        ```

        As with all polymorphic functions, no construction is actually performed
        until invocation, and at invocation time, the default parameter values
        are used alongside those actually used during the invocation to infer
        the computation's parameter type. The code that actually constructs the
        computation is oblivious to whether parameters of the Python function
        being decorated were driven by the default values, or by the arguments
        of the actual call.

        Note that the last argument to the function in the example above will
        be inferred as type `('name', str)`, not just `str`.

  For more examples of usage, see `computation_wrapper_test`.
  """

  def __init__(
      self,
      wrapper_fn: Callable[..., computation_base.Computation],
      to_type_fn: Callable[
          [object], computation_types.Type
      ] = computation_types.to_type,
      infer_type_fn: Callable[
          [object], computation_types.Type
      ] = type_conversions.infer_type,
  ):
    """Construct a new wrapper/decorator for the given wrapper callable.

    Args:
      wrapper_fn: The Python callable that performs actual wrapping (as in the
        specification of `_wrap`).
      to_type_fn: A `Callable` used to convert a backend-specific type
        specification to a `federated_language.Type`.
      infer_type_fn: A `Callable` used to convert a backend-specific values to a
        `federated_language.Type`.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    self._wrapper_fn = wrapper_fn
    self._to_type_fn = to_type_fn
    self._infer_type_fn = infer_type_fn

  def __call__(self, *args, **kwargs):
    """Handles the different modes of usage of the decorator/wrapper.

    This method only acts as a frontend that allows this class to be used as a
    decorator or wrapper in a variety of ways. The actual wrapping is performed
    by the private method `_wrap`.

    Args:
      *args: Positional arguments for the computation decorator.
      **kwargs: Keyword arguments passed to individual computation strategies.

    Returns:
      Either a result of wrapping, or a callable that expects a function,
      method, or a tf.function and performs wrapping on it, depending on
      specific usage pattern.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """

    def _to_types(
        objs: Iterable[object],
    ) -> tuple[Optional[computation_types.Type], ...]:
      result = []
      for obj in objs:
        if obj is not None:
          result.append(self._to_type_fn(obj))
        else:
          result.append(None)
      return tuple(result)

    if not args:
      # If invoked as a decorator, and with an empty argument list as "@xyz()"
      # applied to a function definition, expect the Python function being
      # decorated to be passed in the subsequent call, and potentially create
      # a polymorphic callable. The parameter type is unspecified.
      # Deliberate wrapping with a lambda to prevent the caller from being able
      # to accidentally specify parameter type as a second argument.
      # The tricky partial recursion is needed to inline the logic in the
      # "success" case below.
      provided_types = []
      return lambda fn: _wrap(
          fn, self._wrapper_fn, provided_types, self._infer_type_fn
      )
    elif _is_function(args[0]):
      # If the first argument on the list is a Python function, instance method,
      # or a tf.function, this is the one that's being wrapped. This is the case
      # of either a decorator invocation without arguments as "@xyz" applied to
      # a function definition, of an inline invocation as
      # `... = xyz(lambda....).`
      # Any of the following arguments, if present, are the arguments to the
      # wrapper that are to be interpreted as the type specification.
      fn = args[0]
      provided_types = _to_types(args[1:])
      return _wrap(fn, self._wrapper_fn, provided_types, self._infer_type_fn)
    else:
      provided_types = _to_types(args)
      return lambda fn: _wrap(
          fn, self._wrapper_fn, provided_types, self._infer_type_fn
      )
