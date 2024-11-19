load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:private"],
)

license(
    name = "package_license",
    package_name = "federated_language",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
    visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
    "README.md",
    "requirements_lock_3_10.txt",
    "requirements_lock_3_11.txt",
    "requirements_lock_3_9.txt",
    "requirements.in",
])

filegroup(
    name = "pyproject_toml",
    srcs = ["pyproject.toml"],
    visibility = ["//tools:__subpackages__"],
)
