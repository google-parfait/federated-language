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
        "//federated_language/executors:__subpackages__",
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "federated_context",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(name = "data")

py_library(
    name = "federated_computation",
    srcs = ["federated_computation.py"],
    deps = [
        ":federated_computation_utils",
        "//federated_language/computation:computation_impl",
        "//federated_language/computation:computation_wrapper",
        "//federated_language/computation:function_utils",
        "//federated_language/context_stack:context_stack_impl",
    ],
)

py_test(
    name = "federated_computation_test",
    size = "small",
    srcs = ["federated_computation_test.py"],
    deps = [
        ":federated_computation",
        "//federated_language/computation:computation_impl",
        "//federated_language/computation:computation_wrapper",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/context_stack:runtime_error_context",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "federated_computation_context",
    srcs = ["federated_computation_context.py"],
    deps = [
        ":value_impl",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/compiler:building_blocks",
        "//federated_language/context_stack:context_stack_base",
        "//federated_language/context_stack:symbol_binding_context",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_conversions",
    ],
)

py_test(
    name = "federated_computation_context_test",
    size = "small",
    srcs = ["federated_computation_context_test.py"],
    deps = [
        ":federated_computation",
        ":federated_computation_context",
        ":value_impl",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:computation_factory",
        "//federated_language/computation:computation_impl",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "federated_computation_utils",
    srcs = ["federated_computation_utils.py"],
    deps = [
        ":federated_computation_context",
        ":value_impl",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/compiler:building_blocks",
        "//federated_language/computation:computation_wrapper",
        "//federated_language/context_stack:context_stack_base",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_conversions",
    ],
)

py_test(
    name = "federated_computation_utils_test",
    size = "small",
    srcs = ["federated_computation_utils_test.py"],
    deps = [
        ":federated_computation_utils",
        "//federated_language/computation:computation_wrapper",
        "//federated_language/computation:function_utils",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "intrinsics",
    srcs = ["intrinsics.py"],
    deps = [
        ":value_impl",
        ":value_utils",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/compiler:building_block_factory",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:intrinsic_defs",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/context_stack:symbol_binding_context",
        "//federated_language/types:array_shape",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
        "//federated_language/types:type_factory",
        "@pypi//numpy",
    ],
)

py_test(
    name = "intrinsics_test",
    srcs = ["intrinsics_test.py"],
    deps = [
        ":federated_computation_context",
        ":intrinsics",
        ":value_impl",
        "//federated_language/compiler:building_blocks",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/context_stack:context_stack_test_utils",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "value_impl",
    srcs = ["value_impl.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/compiler:array",
        "//federated_language/compiler:building_block_factory",
        "//federated_language/compiler:building_blocks",
        "//federated_language/computation:computation_impl",
        "//federated_language/computation:function_utils",
        "//federated_language/computation:polymorphic_computation",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/context_stack:symbol_binding_context",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_conversions",
        "//federated_language/types:typed_object",
        "@pypi//attrs",
    ],
)

py_test(
    name = "value_impl_test",
    size = "small",
    srcs = ["value_impl_test.py"],
    deps = [
        ":federated_computation_context",
        ":value_impl",
        "//federated_language/common_libs:structure",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:computation_factory",
        "//federated_language/computation:computation_impl",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//numpy",
    ],
)

py_library(
    name = "value_utils",
    srcs = ["value_utils.py"],
    deps = [
        ":value_impl",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/compiler:building_block_factory",
        "//federated_language/compiler:building_blocks",
        "//federated_language/types:computation_types",
    ],
)

py_test(
    name = "value_utils_test",
    size = "small",
    srcs = ["value_utils_test.py"],
    deps = [
        ":federated_computation",
        ":federated_computation_context",
        ":value_impl",
        ":value_utils",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:computation_factory",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)
