load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//federated_language:__pkg__",
        "//federated_language/execution_contexts:__subpackages__",
        "//federated_language/framework:__subpackages__",
        "//federated_language/program:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "executors",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "cardinalities_utils",
    srcs = ["cardinalities_utils.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/common_libs:structure",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
    ],
)

py_test(
    name = "cardinalities_utils_test",
    srcs = ["cardinalities_utils_test.py"],
    deps = [
        ":cardinalities_utils",
        "//federated_language/common_libs:structure",
        "//federated_language/types:computation_types",
        "//federated_language/types:placements",
        "@pypi//absl_py",
        "@pypi//numpy",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    deps = [":executor_value_base"],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    deps = [
        ":executor_base",
        "//federated_language/types:placements",
    ],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    deps = ["//federated_language/types:typed_object"],
)

py_library(
    name = "executors_errors",
    srcs = ["executors_errors.py"],
)
