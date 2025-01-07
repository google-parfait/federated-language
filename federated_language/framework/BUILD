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

load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = [
        "//federated_language:__pkg__",
        "//tools:__subpackages__",
    ],
    deps = [
        "//federated_language/common_libs:tracing",
        "//federated_language/compiler:building_block_analysis",
        "//federated_language/compiler:building_block_factory",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:intrinsic_defs",
        "//federated_language/compiler:transformation_utils",
        "//federated_language/compiler:tree_analysis",
        "//federated_language/computation:computation_base",
        "//federated_language/computation:computation_impl",
        "//federated_language/computation:computation_wrapper",
        "//federated_language/computation:function_utils",
        "//federated_language/computation:polymorphic_computation",
        "//federated_language/context_stack:context_base",
        "//federated_language/context_stack:context_stack_base",
        "//federated_language/context_stack:context_stack_impl",
        "//federated_language/context_stack:context_stack_test_utils",
        "//federated_language/context_stack:get_context_stack",
        "//federated_language/context_stack:runtime_error_context",
        "//federated_language/context_stack:set_default_context",
        "//federated_language/context_stack:symbol_binding_context",
        "//federated_language/execution_contexts:async_execution_context",
        "//federated_language/execution_contexts:compiler_pipeline",
        "//federated_language/execution_contexts:sync_execution_context",
        "//federated_language/executors:cardinalities_utils",
        "//federated_language/executors:executor_base",
        "//federated_language/executors:executor_factory",
        "//federated_language/executors:executor_value_base",
        "//federated_language/executors:executors_errors",
        "//federated_language/federated_context:federated_computation_context",
        "//federated_language/federated_context:value_utils",
        "//federated_language/test:static_assert",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_analysis",
        "//federated_language/types:type_conversions",
        "//federated_language/types:type_test_utils",
        "//federated_language/types:type_transformations",
    ],
)
