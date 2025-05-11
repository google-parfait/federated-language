# Copyright 2024 Google LLC
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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    visibility = ["//visibility:private"],
)

proto_library(
    name = "array_proto",
    srcs = ["array.proto"],
    deps = [":data_type_proto"],
)

py_proto_library(
    name = "array_py_pb2",
    deps = [":array_proto"],
)

cc_proto_library(
    name = "array_cc_proto",
    deps = [":array_proto"],
)

proto_library(
    name = "computation_proto",
    srcs = ["computation.proto"],
    deps = [
        ":array_proto",
        ":data_type_proto",
        "@protobuf//:any_proto",
    ],
)

py_proto_library(
    name = "computation_py_pb2",
    deps = [":computation_proto"],
)

cc_proto_library(
    name = "computation_cc_proto",
    deps = [":computation_proto"],
)

proto_library(
    name = "data_type_proto",
    srcs = ["data_type.proto"],
)

py_proto_library(
    name = "data_type_py_pb2",
    deps = [":data_type_proto"],
)

cc_proto_library(
    name = "data_type_cc_proto",
    deps = [":data_type_proto"],
)
