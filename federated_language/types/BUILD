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
        "//federated_language/compiler:__subpackages__",
        "//federated_language/computation:__subpackages__",
        "//federated_language/executor:__subpackages__",
        "//federated_language/federated_context:__subpackages__",
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    visibility = ["//visibility:private"],
)

py_library(
    name = "array_shape",
    srcs = ["array_shape.py"],
    deps = [
        "//federated_language/proto:array_py_pb2",
        "//federated_language/proto:data_type_py_pb2",
    ],
)

py_test(
    name = "array_shape_test",
    srcs = ["array_shape_test.py"],
    deps = [
        ":array_shape",
        "//federated_language/proto:array_py_pb2",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "computation_types",
    srcs = ["computation_types.py"],
    deps = [
        ":array_shape",
        ":dtype_utils",
        ":placements",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/proto:array_py_pb2",
        "//federated_language/proto:computation_py_pb2",
        "@federated_language_pypi//attrs",
        "@federated_language_pypi//numpy",
        "@federated_language_pypi//typing_extensions",
    ],
)

py_test(
    name = "computation_types_test",
    size = "small",
    srcs = ["computation_types_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        "//federated_language/common_libs:structure",
        "//federated_language/proto:computation_py_pb2",
        "//federated_language/proto:data_type_py_pb2",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//attrs",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "dtype_utils",
    srcs = ["dtype_utils.py"],
    deps = [
        "//federated_language/proto:data_type_py_pb2",
        "@federated_language_pypi//ml_dtypes",
        "@federated_language_pypi//numpy",
    ],
)

py_test(
    name = "dtype_utils_test",
    srcs = ["dtype_utils_test.py"],
    deps = [
        ":dtype_utils",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//ml_dtypes",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
)

py_test(
    name = "placements_test",
    size = "small",
    srcs = ["placements_test.py"],
    deps = [
        ":placements",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "type_analysis",
    srcs = ["type_analysis.py"],
    deps = [
        ":array_shape",
        ":computation_types",
        ":placements",
        "@federated_language_pypi//ml_dtypes",
        "@federated_language_pypi//numpy",
    ],
)

py_test(
    name = "type_analysis_test",
    size = "small",
    srcs = ["type_analysis_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_analysis",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//ml_dtypes",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "type_conversions",
    srcs = ["type_conversions.py"],
    deps = [
        ":computation_types",
        ":dtype_utils",
        ":typed_object",
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "@federated_language_pypi//attrs",
        "@federated_language_pypi//dm_tree",
        "@federated_language_pypi//numpy",
    ],
)

py_test(
    name = "type_conversions_test",
    size = "small",
    srcs = ["type_conversions_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_conversions",
        ":typed_object",
        "//federated_language/common_libs:structure",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//attrs",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "type_factory",
    srcs = ["type_factory.py"],
    deps = [":computation_types"],
)

py_test(
    name = "type_factory_test",
    size = "small",
    srcs = ["type_factory_test.py"],
    deps = [
        ":computation_types",
        ":type_factory",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "type_transformations",
    srcs = ["type_transformations.py"],
    deps = [":computation_types"],
)

py_test(
    name = "type_transformations_test",
    size = "small",
    srcs = ["type_transformations_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_transformations",
        "@federated_language_pypi//absl_py",
        "@federated_language_pypi//numpy",
    ],
)

py_library(
    name = "typed_object",
    srcs = ["typed_object.py"],
    deps = [":computation_types"],
)
