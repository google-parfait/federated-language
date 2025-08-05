# Copyright 2025 Google LLC
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

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["@federated_language//:package_license"],
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "computations",
    srcs = ["computations.py"],
    deps = [
        "//computation:python_computation",
        "@examples_pypi//numpy",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    deps = [
        ":computations",
        "//algorithm:program_logic",
        "//executor:execution_context",
        "//federated_platform:data_source",
        "//federated_platform:program_state_manager",
        "@examples_pypi//absl_py",
        "@federated_language//federated_language",
    ],
)

py_binary(
    name = "program",
    srcs = ["program.py"],
    deps = [
        ":computations",
        "//algorithm:program_logic",
        "//executor:execution_context",
        "//federated_platform:data_source",
        "//federated_platform:program_state_manager",
        "@examples_pypi//absl_py",
        "@federated_language//federated_language",
    ],
)
