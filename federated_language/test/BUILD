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
    default_visibility = ["//federated_language/framework:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "static_assert",
    srcs = ["static_assert.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:tree_analysis",
        "//federated_language/computation:computation_impl",
    ],
)

py_test(
    name = "static_assert_test",
    size = "small",
    srcs = ["static_assert_test.py"],
    deps = [
        ":static_assert",
        "//federated_language/federated_context:federated_computation",
        "//federated_language/federated_context:intrinsics",
        "//federated_language/types:placements",
        "@pypi//absl_py",
    ],
)
