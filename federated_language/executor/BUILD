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
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "executor",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "async_execution_context",
    srcs = ["async_execution_context.py"],
    deps = [
        "//federated_language/common_libs:retrying",
        "//federated_language/common_libs:structure",
        "//federated_language/common_libs:tracing",
        "//federated_language/computation:computation_base",
        "//federated_language/computation:function_utils",
        "//federated_language/context_stack:context",
        "//federated_language/executor:cardinalities_utils",
        "//federated_language/executor:executor_base",
        "//federated_language/executor:executor_factory",
        "//federated_language/executor:executor_value_base",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "//federated_language/types:type_conversions",
        "@federated_language_pypi//dm_tree",
    ],
)

py_test(
    name = "async_execution_context_test",
    size = "small",
    srcs = ["async_execution_context_test.py"],
    deps = [
        ":async_execution_context",
        "//federated_language/executor:executor_base",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "cardinalities_utils",
    srcs = ["cardinalities_utils.py"],
    deps = [
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
    ],
)

py_test(
    name = "cardinalities_utils_test",
    srcs = ["cardinalities_utils_test.py"],
    deps = [
        ":cardinalities_utils",
        "//federated_language/common_libs:structure",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    deps = [":executor_value_base"],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    deps = [
        ":executor_base",
        "//federated_language/types:placements",
    ],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    deps = ["//federated_language/types:typed_object"],
)

py_library(
    name = "sync_execution_context",
    srcs = ["sync_execution_context.py"],
    deps = [
        ":async_execution_context",
        "//federated_language/common_libs:async_utils",
        "//federated_language/computation:computation_base",
        "//federated_language/context_stack:context",
        "//federated_language/executor:cardinalities_utils",
        "//federated_language/executor:executor_factory",
    ],
)
