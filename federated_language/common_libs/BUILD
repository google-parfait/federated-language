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
    default_visibility = ["//federated_language:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "common_libs",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "async_utils",
    srcs = ["async_utils.py"],
    deps = [":tracing"],
)

py_library(
    name = "golden",
    testonly = True,
    srcs = ["golden.py"],
    deps = ["@federated_language_pypi//absl_py"],
)

py_test(
    name = "golden_test",
    size = "small",
    srcs = ["golden_test.py"],
    args = [
        "--golden",
        "$(location golden_test_goldens/test_check_string_fails.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_succeeds.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_updates.expected)",
    ],
    data = [
        "golden_test_goldens/test_check_string_fails.expected",
        "golden_test_goldens/test_check_string_succeeds.expected",
        "golden_test_goldens/test_check_string_updates.expected",
    ],
    deps = [
        ":golden",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    deps = ["@federated_language_pypi//typing_extensions"],
)

py_test(
    name = "py_typecheck_test",
    size = "small",
    srcs = ["py_typecheck_test.py"],
    deps = [
        ":py_typecheck",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "retrying",
    srcs = ["retrying.py"],
)

py_test(
    name = "retrying_test",
    size = "small",
    srcs = ["retrying_test.py"],
    deps = [
        ":retrying",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "serializable",
    srcs = ["serializable.py"],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        ":py_typecheck",
        "@federated_language_pypi//attrs",
        "@federated_language_pypi//dm_tree",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    deps = [
        ":structure",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//attrs",
    ],
)

py_library(
    name = "tracing",
    srcs = ["tracing.py"],
    deps = ["@federated_language_pypi//absl_py"],
)

py_test(
    name = "tracing_test",
    size = "small",
    srcs = ["tracing_test.py"],
    deps = [
        ":tracing",
        "@federated_language_pypi//absl_py",
    ],
)
