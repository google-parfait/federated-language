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
    name = "federated_language",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//third_party/py/federated_language/common_libs:serializable",
        "//third_party/py/federated_language/compiler:array",
        "//third_party/py/federated_language/computation:computation_base",
        "//third_party/py/federated_language/federated_context:federated_computation",
        "//third_party/py/federated_language/federated_context:intrinsics",
        "//third_party/py/federated_language/federated_context:value_impl",
        "//third_party/py/federated_language/framework",
        "//third_party/py/federated_language/program",
        "//third_party/py/federated_language/types:array_shape",
        "//third_party/py/federated_language/types:computation_types",
        "//third_party/py/federated_language/types:dtype_utils",
        "//third_party/py/federated_language/types:placements",
        "//third_party/py/federated_language/types:typed_object",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
