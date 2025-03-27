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
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_library_providing_imports_info")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library_providing_imports_info(
    name = "federated_language",
    srcs = ["__init__.py"],
    lib_rule = pytype_strict_library,
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//federated_language/common_libs:serializable",
        "//federated_language/compiler:array",
        "//federated_language/computation:computation_base",
        "//federated_language/federated_context:federated_computation",
        "//federated_language/federated_context:intrinsics",
        "//federated_language/federated_context:value_impl",
        "//federated_language/framework",
        "//federated_language/program",
        "//federated_language/types:array_shape",
        "//federated_language/types:computation_types",
        "//federated_language/types:dtype_utils",
        "//federated_language/types:placements",
        "//federated_language/types:typed_object",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
