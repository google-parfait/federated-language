# Copyright 2025 Google LLC
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
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_applicable_licenses = ["@federated_language//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    deps = [
        "@examples_pypi//cloudpickle",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:array_py_pb2",
    ],
)

py_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.py"],
    deps = [
        ":execution_context",
        ":python_computation",
        "@examples_pypi//absl_py",
        "@examples_pypi//numpy",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language/proto:data_type_py_pb2",
    ],
)

py_library(
    name = "python_computation",
    srcs = ["python_computation.py"],
    deps = [
        "@examples_pypi//cloudpickle",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@protobuf//:protobuf_python",
    ],
)

py_test(
    name = "python_computation_test",
    srcs = ["python_computation_test.py"],
    deps = [
        ":python_computation",
        "@examples_pypi//absl_py",
        "@examples_pypi//numpy",
        "@federated_language//federated_language",
    ],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
