# Copyright 2018 Google LLC
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
"""A library of (de)serialization functions for computation types."""

from federated_language.proto import computation_pb2
from federated_language.types import computation_types


def serialize_type(type_spec: computation_types.Type) -> computation_pb2.Type:
  return type_spec.to_proto()


def deserialize_type(
    type_proto: computation_pb2.Type,
) -> computation_types.Type:
  return computation_types.Type.from_proto(type_proto)
