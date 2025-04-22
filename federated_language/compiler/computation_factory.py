# Copyright 2020 Google LLC
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
"""A library of construction functions for computation structures."""

from federated_language.proto import computation_pb2
from federated_language.types import computation_types
from federated_language.types import type_factory


def create_lambda_empty_struct() -> computation_pb2.Computation:
  """Returns a lambda computation returning an empty struct.

  Has the type signature:

  ( -> <>)

  Returns:
    An instance of `computation_pb2.Computation`.
  """
  result_type = computation_types.StructType([])
  type_signature = computation_types.FunctionType(None, result_type)
  result = computation_pb2.Computation(
      type=result_type.to_proto(),
      struct=computation_pb2.Struct(element=[]),
  )
  fn = computation_pb2.Lambda(parameter_name=None, result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `computation_pb2.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  return computation_pb2.Computation(
      type=type_signature.to_proto(), **{'lambda': fn}
  )  # pytype: disable=wrong-keyword-args


def create_lambda_identity(
    type_spec: computation_types.Type,
) -> computation_pb2.Computation:
  """Returns a lambda computation representing an identity function.

  Has the type signature:

  (T -> T)

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    An instance of `computation_pb2.Computation`.
  """
  type_signature = type_factory.unary_op(type_spec)
  result = computation_pb2.Computation(
      type=type_spec.to_proto(),
      reference=computation_pb2.Reference(name='a'),
  )
  fn = computation_pb2.Lambda(parameter_name='a', result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `computation_pb2.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  return computation_pb2.Computation(
      type=type_signature.to_proto(), **{'lambda': fn}
  )  # pytype: disable=wrong-keyword-args
