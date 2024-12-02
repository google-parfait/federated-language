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
