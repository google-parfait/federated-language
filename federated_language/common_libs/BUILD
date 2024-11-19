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
    deps = ["@pypi//absl_py"],
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
        "@pypi//absl_py",
    ],
)

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    deps = [
        "@pypi//attrs",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "py_typecheck_test",
    size = "small",
    srcs = ["py_typecheck_test.py"],
    deps = [
        ":py_typecheck",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "retrying",
    srcs = ["retrying.py"],
    deps = [
        ":py_typecheck",
    ],
)

py_test(
    name = "retrying_test",
    size = "small",
    srcs = ["retrying_test.py"],
    deps = [
        ":retrying",
        "@pypi//absl_py",
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
        "@pypi//attrs",
        "@pypi//dm_tree",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    deps = [
        ":structure",
        "@pypi//absl_py",
        "@pypi//attrs",
    ],
)

py_library(
    name = "tracing",
    srcs = ["tracing.py"],
    deps = [
        ":py_typecheck",
        "@pypi//absl_py",
    ],
)

py_test(
    name = "tracing_test",
    size = "small",
    srcs = ["tracing_test.py"],
    deps = [
        ":tracing",
        "@pypi//absl_py",
    ],
)
