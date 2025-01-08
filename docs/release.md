# Release

1.  Choose a version (using the [Semantic Versioning](https://semver.org/)
    scheme).

    Given a version number `MAJOR.MINOR.PATCH`, increment the:

    *   Don't increment the **MAJOR** version, this project is in an
        [initial development phase](https://semver.org/#spec-item-4).
    *   Increment the the **MINOR** version when making incompatible API changes
        or adding functionality in a backward compatible manner.
    *   Increment the **PATCH** version when making backward compatible bug
        fixes.

    Tip: It may be helpful to
    [compare](https://github.com/google-parfait/federated-language/compare/) the
    latest release and the `main` branch to see the commits since the latest
    release.

1.  Submit a CL to update the version in
    [version.py](https://github.com/google-parfait/federated-language/blob/main/federated_language/version.py)
    and
    [MODULE.bazel](https://github.com/google-parfait/federated-language/blob/main/federated_language/MODULE.bazel).

1.  Create and push a tag to the
    [federated-language](https://github.com/google-parfait/federated-language/)
    repository.

    Important: Requires creating a
    [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

    ```shell
    $ git clone \
        https://<USERNAME>:<TOKEN>@github.com/google-parfait/federated-language/ \
        "/tmp/federated-language"
    $ cd "/tmp/federated-language"
    $ git checkout <COMMIT>
    $ git tag v<VERSION>
    $ git push origin tag v<VERSION>
    ```

1.  Wait for the
    [release](https://github.com/google-parfait/federated-language/actions/workflows/release.yaml)
    workflow.

    *   Publish a package to
        [PyPI](https://pypi.org/project/federated-language/#history).
    *   Publish a release to
        [GitHub](https://github.com/google-parfait/federated-language/releases).
    *   Create a pull request (e.g.,
        [#3541](https://github.com/bazelbuild/bazel-central-registry/pull/3541))
        to publish a module to
        [BCR](https://registry.bazel.build/modules/federated_language).

1.  Approve the pull request.

1.  Wait for the pull request to be merged.
