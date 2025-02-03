# Copyright 2024 Google LLC
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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//federated_language:__pkg__",
        "//federated_language/computation:__subpackages__",
        "//federated_language/execution_contexts:__subpackages__",
        "//federated_language/executors:__subpackages__",
        "//federated_language/federated_context:__subpackages__",
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "array",
    srcs = ["array.py"],
    deps = [
        "//federated_language/proto:array_py_pb2",
        "//federated_language/types:array_shape",
        "//federated_language/types:dtype_utils",
        "@pypi//ml_dtypes",
        "@pypi//numpy",
    ],
)

py_test(
    name = "array_test",
    srcs = ["array_test.py"],
    deps = [
        ":array",
        "//federated_language/proto:array_py_pb2",
        "//federated_language/proto:data_type_py_pb2",
        "@pypi//absl_py",
        "@pypi//ml_dtypes",
        "@pypi//numpy",
    ],
)

py_library(
    name = "building_block_analysis",
    srcs = ["building_block_analysis.py"],
    deps = [
        ":building_blocks",
        ":intrinsic_defs",
    ],
)

py_library(
    name = "building_block_factory",
    srcs = ["building_block_factory.py"],
    deps = [
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_transformations",
    ],
)

py_test(
    name = "building_block_factory_test",
    size = "large",
    srcs = ["building_block_factory_test.py"],
    args = [
        "--golden",
        "$(location building_block_factory_test_goldens/constructs_correct_computation_clients.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/constructs_correct_computation_server.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/replaces_single_element.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/skips_unnamed_element.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_map_with_two_values_different_typed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/tuple_federated_apply_with_two_values_different_typed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_tuple_unnamed.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_tuple_named.expected)",
        "--golden",
        "$(location building_block_factory_test_goldens/zips_reference.expected)",
    ],
    data = [
        "building_block_factory_test_goldens/constructs_correct_computation_clients.expected",
        "building_block_factory_test_goldens/constructs_correct_computation_server.expected",
        "building_block_factory_test_goldens/replaces_single_element.expected",
        "building_block_factory_test_goldens/skips_unnamed_element.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_different_typed.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_named.expected",
        "building_block_factory_test_goldens/tuple_federated_apply_with_two_values_unnamed.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_different_typed.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_named.expected",
        "building_block_factory_test_goldens/tuple_federated_map_with_two_values_unnamed.expected",
        "building_block_factory_test_goldens/zips_reference.expected",
        "building_block_factory_test_goldens/zips_tuple_named.expected",
        "building_block_factory_test_goldens/zips_tuple_unnamed.expected",
    ],
    deps = [
        ":building_block_factory",
        ":building_blocks",
        "//federated_language/common_libs:golden",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "building_blocks",
    srcs = ["building_blocks.py"],
    deps = [
        ":array",
        ":intrinsic_defs",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
        "//federated_language/types:typed_object",
        "@protobuf//:protobuf_python",
        "@pypi//numpy",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "building_blocks_test",
    size = "small",
    srcs = ["building_blocks_test.py"],
    deps = [
        ":array",
        ":building_block_factory",
        ":building_blocks",
        ":computation_factory",
        ":intrinsic_defs",
        "//federated_language/common_libs:structure",
        "//federated_language/proto:array_py_pb2",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/proto:data_type_py_pb2",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@protobuf//:protobuf_python",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//ml_dtypes",
        "@pypi//numpy",
    ],
)

py_library(
    name = "computation_factory",
    srcs = ["computation_factory.py"],
    deps = [
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_factory",
    ],
)

py_test(
    name = "computation_factory_test",
    srcs = ["computation_factory_test.py"],
    deps = [
        ":computation_factory",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_factory",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_factory",
        "@pypi//numpy",
    ],
)

py_test(
    name = "intrinsic_defs_test",
    size = "small",
    srcs = ["intrinsic_defs_test.py"],
    deps = [
        ":intrinsic_defs",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "transformation_utils",
    srcs = ["transformation_utils.py"],
    deps = [
        ":building_blocks",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
    ],
)

py_test(
    name = "transformation_utils_test",
    size = "small",
    srcs = ["transformation_utils_test.py"],
    deps = [
        ":building_blocks",
        ":computation_factory",
        ":transformation_utils",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@protobuf//:protobuf_python",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "tree_analysis",
    srcs = ["tree_analysis.py"],
    deps = [
        ":building_block_analysis",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
    ],
)

py_test(
    name = "tree_analysis_test",
    srcs = ["tree_analysis_test.py"],
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":tree_analysis",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)
