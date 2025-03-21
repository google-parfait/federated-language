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
        "//federated_language/execution_contexts:__subpackages__",
        "//federated_language/executors:__subpackages__",
        "//federated_language/federated_context:__subpackages__",
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "computation",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "computation_analysis",
    srcs = ["computation_analysis.py"],
    deps = [
        ":computation_impl",
        "//federated_language/compiler:building_block_analysis",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:intrinsic_defs",
        "//federated_language/compiler:tree_analysis",
    ],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
    deps = [
        "//federated_language/types:computation_types",
        "//federated_language/types:typed_object",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        ":computation_base",
        ":function_utils",
        "//federated_language/compiler:building_blocks",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
    ],
)

py_test(
    name = "computation_impl_test",
    size = "small",
    srcs = ["computation_impl_test.py"],
    deps = [
        ":computation_impl",
        "//federated_language/compiler:building_blocks",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(name = "computation_serialization")

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    deps = [
        ":computation_base",
        ":computation_impl",
        ":polymorphic_computation",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_conversions",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    deps = [
        ":computation_impl",
        ":computation_wrapper",
        ":function_utils",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_conversions",
        "//federated_language/types:typed_object",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    deps = [
        ":function_utils",
        "//federated_language/common_libs:structure",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "polymorphic_computation",
    srcs = ["polymorphic_computation.py"],
    deps = [
        ":computation_impl",
        ":function_utils",
        "//federated_language/types:computation_types",
    ],
)

py_test(
    name = "polymorphic_computation_test",
    srcs = ["polymorphic_computation_test.py"],
    deps = [
        ":computation_impl",
        ":polymorphic_computation",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
        "//federated_language/types:type_conversions",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)
