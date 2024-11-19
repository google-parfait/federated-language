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
        "//:pyproject_toml",
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
        "//federated_language/test",
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
