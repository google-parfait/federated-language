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
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "program",
    srcs = ["__init__.py"],
    visibility = ["//federated_language:__pkg__"],
    deps = [
        ":data_source",
        ":federated_context",
        ":logging_release_manager",
        ":memory_release_manager",
        ":native_platform",
        ":program_state_manager",
        ":release_manager",
        ":value_reference",
    ],
)

py_library(
    name = "data_source",
    srcs = ["data_source.py"],
    deps = [
        "//federated_language/common_libs:serializable",
        "//federated_language/types:computation_types",
    ],
)

py_library(
    name = "federated_context",
    srcs = ["federated_context.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//federated_language/computation:computation_base",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
    ],
)

py_test(
    name = "federated_context_test",
    srcs = ["federated_context_test.py"],
    deps = [
        ":federated_context",
        "//federated_language/context_stack:context",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "logging_release_manager",
    srcs = ["logging_release_manager.py"],
    deps = [
        ":release_manager",
        ":value_reference",
        "@pypi//absl_py",
    ],
)

py_test(
    name = "logging_release_manager_test",
    srcs = ["logging_release_manager_test.py"],
    deps = [
        ":logging_release_manager",
        ":program_test_utils",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "memory_release_manager",
    srcs = ["memory_release_manager.py"],
    deps = [
        ":release_manager",
        ":value_reference",
    ],
)

py_test(
    name = "memory_release_manager_test",
    srcs = ["memory_release_manager_test.py"],
    deps = [
        ":memory_release_manager",
        ":program_test_utils",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "native_platform",
    srcs = ["native_platform.py"],
    deps = [
        ":federated_context",
        ":structure_utils",
        ":value_reference",
        "//federated_language/common_libs:structure",
        "//federated_language/computation:computation_base",
        "//federated_language/execution_contexts:async_execution_context",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_conversions",
        "@pypi//dm_tree",
    ],
)

py_test(
    name = "native_platform_test",
    srcs = ["native_platform_test.py"],
    deps = [
        ":native_platform",
        ":program_test_utils",
        ":structure_utils",
        ":value_reference",
        "//federated_language/computation:computation_impl",
        "//federated_language/execution_contexts:async_execution_context",
        "//federated_language/federated_context:federated_computation",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "program_state_manager",
    srcs = ["program_state_manager.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//federated_language/common_libs:serializable",
    ],
)

py_test(
    name = "program_state_manager_test",
    srcs = ["program_state_manager_test.py"],
    deps = [
        ":program_state_manager",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "program_test_utils",
    testonly = True,
    srcs = ["program_test_utils.py"],
    deps = [
        ":value_reference",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:serializable",
        "//federated_language/types:computation_types",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "release_manager",
    srcs = ["release_manager.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//federated_language/common_libs:py_typecheck",
        "@pypi//attrs",
        "@pypi//dm_tree",
    ],
)

py_test(
    name = "release_manager_test",
    srcs = ["release_manager_test.py"],
    deps = [
        ":program_test_utils",
        ":release_manager",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    deps = [
        "//federated_language/common_libs:serializable",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/types:computation_types",
    ],
)

py_test(
    name = "serialization_utils_test",
    srcs = ["serialization_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":serialization_utils",
        "//federated_language/types:computation_types",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "@pypi//attrs",
        "@pypi//dm_tree",
    ],
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":structure_utils",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)

py_library(
    name = "value_reference",
    srcs = ["value_reference.py"],
    deps = [
        ":structure_utils",
        "//federated_language/types:computation_types",
        "//federated_language/types:typed_object",
        "@pypi//numpy",
    ],
)

py_test(
    name = "value_reference_test",
    srcs = ["value_reference_test.py"],
    deps = [
        ":program_test_utils",
        ":value_reference",
        "@pypi//absl_py",
        "@pypi//dm_tree",
        "@pypi//numpy",
    ],
)
