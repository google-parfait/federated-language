# Copyright 2019 Google LLC
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
"""The Federated Language library."""


# pylint: disable=g-importing-member
from federated_language import framework
from federated_language import program
from federated_language.common_libs.serializable import Serializable
from federated_language.compiler.array import Array
from federated_language.compiler.array import from_proto as array_from_proto
from federated_language.compiler.array import from_proto_content as array_from_proto_content
from federated_language.compiler.array import is_compatible_dtype as array_is_compatible_dtype
from federated_language.compiler.array import is_compatible_shape as array_is_compatible_shape
from federated_language.compiler.array import to_proto as array_to_proto
from federated_language.compiler.array import to_proto_content as array_to_proto_content
from federated_language.computation.computation_base import Computation
from federated_language.federated_context.federated_computation import federated_computation
from federated_language.federated_context.intrinsics import federated_aggregate
from federated_language.federated_context.intrinsics import federated_broadcast
from federated_language.federated_context.intrinsics import federated_eval
from federated_language.federated_context.intrinsics import federated_map
from federated_language.federated_context.intrinsics import federated_max
from federated_language.federated_context.intrinsics import federated_mean
from federated_language.federated_context.intrinsics import federated_min
from federated_language.federated_context.intrinsics import federated_secure_select
from federated_language.federated_context.intrinsics import federated_secure_sum
from federated_language.federated_context.intrinsics import federated_secure_sum_bitwidth
from federated_language.federated_context.intrinsics import federated_select
from federated_language.federated_context.intrinsics import federated_sum
from federated_language.federated_context.intrinsics import federated_value
from federated_language.federated_context.intrinsics import federated_zip
from federated_language.federated_context.intrinsics import sequence_map
from federated_language.federated_context.intrinsics import sequence_reduce
from federated_language.federated_context.intrinsics import sequence_sum
from federated_language.federated_context.value_impl import to_value
from federated_language.federated_context.value_impl import Value
from federated_language.types.array_shape import ArrayShape
from federated_language.types.array_shape import from_proto as array_shape_from_proto
from federated_language.types.array_shape import is_compatible_with as array_shape_is_compatible_with
from federated_language.types.array_shape import is_shape_fully_defined as array_shape_is_fully_defined
from federated_language.types.array_shape import is_shape_scalar as array_shape_is_scalar
from federated_language.types.array_shape import num_elements_in_shape as num_elements_in_array_shape
from federated_language.types.array_shape import to_proto as array_shape_to_proto
from federated_language.types.computation_types import AbstractType
from federated_language.types.computation_types import FederatedType
from federated_language.types.computation_types import FunctionType
from federated_language.types.computation_types import PlacementType
from federated_language.types.computation_types import SequenceType
from federated_language.types.computation_types import StructType
from federated_language.types.computation_types import StructWithPythonType
from federated_language.types.computation_types import TensorType
from federated_language.types.computation_types import to_type
from federated_language.types.computation_types import Type
from federated_language.types.dtype_utils import can_cast as can_cast_dtype
from federated_language.types.dtype_utils import from_proto as dtype_from_proto
from federated_language.types.dtype_utils import infer_dtype
from federated_language.types.dtype_utils import is_valid_dtype
from federated_language.types.dtype_utils import to_proto as dtype_to_proto
from federated_language.types.placements import CLIENTS
from federated_language.types.placements import SERVER
from federated_language.types.typed_object import TypedObject
from federated_language.version import __version__
# pylint: enable=g-importing-member
