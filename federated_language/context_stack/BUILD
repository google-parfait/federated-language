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
    name = "context_stack",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "context",
    srcs = ["context.py"],
)

py_library(
    name = "context_stack_base",
    srcs = ["context_stack_base.py"],
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
    deps = [
        ":context",
        ":context_stack_base",
        ":runtime_error_context",
        "//federated_language/common_libs:py_typecheck",
    ],
)

py_test(
    name = "context_stack_impl_test",
    size = "small",
    srcs = ["context_stack_impl_test.py"],
    deps = [
        ":context",
        ":context_stack_impl",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "context_stack_test_utils",
    srcs = ["context_stack_test_utils.py"],
    deps = [
        ":context",
        ":context_stack_impl",
    ],
)

py_test(
    name = "context_stack_test_utils_test",
    srcs = ["context_stack_test_utils_test.py"],
    deps = [
        ":context",
        ":context_stack_impl",
        ":context_stack_test_utils",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "get_context_stack",
    srcs = ["get_context_stack.py"],
    deps = [":context_stack_impl"],
)

py_test(
    name = "get_context_stack_test",
    size = "small",
    srcs = ["get_context_stack_test.py"],
    deps = [
        ":context_stack_impl",
        ":get_context_stack",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "runtime_error_context",
    srcs = ["runtime_error_context.py"],
    deps = [":context"],
)

py_library(
    name = "set_default_context",
    srcs = ["set_default_context.py"],
    deps = [
        ":context_stack_impl",
        ":runtime_error_context",
    ],
)

py_test(
    name = "set_default_context_test",
    size = "small",
    srcs = ["set_default_context_test.py"],
    deps = [
        ":context",
        ":context_stack_impl",
        ":set_default_context",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "symbol_binding_context",
    srcs = ["symbol_binding_context.py"],
    deps = [":context"],
)
