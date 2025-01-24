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

load("@python//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
load("@python//3.11:defs.bzl", compile_pip_requirements_3_11 = "compile_pip_requirements")
load("@python//3.9:defs.bzl", compile_pip_requirements_3_9 = "compile_pip_requirements")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

sh_binary(
    name = "build_package",
    srcs = ["build_package.sh"],
    data = [
        "//:LICENSE",
        "//:README.md",
        "//:pyproject.toml",
        "//federated_language",
        "//federated_language/common_libs",
        "//federated_language/compiler",
        "//federated_language/computation",
        "//federated_language/context_stack",
        "//federated_language/execution_contexts",
        "//federated_language/executors",
        "//federated_language/federated_context",
        "//federated_language/framework",
        "//federated_language/program",
        "//federated_language/proto",
        "//federated_language/types",
    ],
)

compile_pip_requirements_3_11(
    name = "requirements_3_11",
    src = "//:requirements.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "//:requirements_lock_3_11.txt",
)

compile_pip_requirements_3_10(
    name = "requirements_3_10",
    src = "//:requirements.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "//:requirements_lock_3_10.txt",
)

compile_pip_requirements_3_9(
    name = "requirements_3_9",
    src = "//:requirements.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "//:requirements_lock_3_9.txt",
)
