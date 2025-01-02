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

1.  Update version in
    [version.py](https://github.com/google-parfait/federated-language/blob/main/federated_language/version.py)
    and
    [MODULE.bazel](https://github.com/google-parfait/federated-language/blob/main/federated_language/MODULE.bazel).

1.  Create and push a tag.

    Important: Requires creating a
    [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

    ```shell
    $ git clone https://<USERNAME>:<TOKEN>@github.com/google-parfait/federated-language/
    $ git tag <VERSION>
    $ git push origin tag <VERSION>
    ```

1.  Wait for the
    [release workflow](https://github.com/google-parfait/federated-language/actions/workflows/release.yaml)
    to publish the release.
