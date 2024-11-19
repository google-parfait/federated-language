load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//federated_language/framework:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tools:__subpackages__"],
)

py_library(
    name = "static_assert",
    srcs = ["static_assert.py"],
    deps = [
        "//federated_language/common_libs:py_typecheck",
        "//federated_language/compiler:building_blocks",
        "//federated_language/compiler:tree_analysis",
        "//federated_language/computation:computation_impl",
    ],
)

py_test(
    name = "static_assert_test",
    size = "small",
    srcs = ["static_assert_test.py"],
    deps = [
        ":static_assert",
        "//federated_language/federated_context:federated_computation",
        "//federated_language/federated_context:intrinsics",
        "//federated_language/types:placements",
        "@pypi//absl_py",
    ],
)
