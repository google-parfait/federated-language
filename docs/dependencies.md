# Dependencies

[TOC]

## Update Python Dependencies

1.  Create a local copy of the change by cloning the repository.

    ```shell
    $ git clone \
        https://github.com/google-parfait/federated-language/ \
        "/tmp/lang"
    $ cd "/tmp/lang"
    ```

1.  Update
    [requirements.in](https://github.com/google-parfait/federated-language/blob/main/federated_language/requirements.in)
    and, if the dependency is required by the Federated Language package, update
    [project.toml](https://github.com/google-parfait/federated-language/blob/main/federated_language/project.toml).

1.  Update the Python requirements lockfiles.

    ```shell
    $ bazelisk run //:requirements_3_9.update -- --upgrade
    $ bazelisk run //:requirements_3_10.update -- --upgrade
    $ bazelisk run //:requirements_3_11.update -- --upgrade
    $ bazelisk run //:requirements_3_12.update -- --upgrade
    $ bazelisk run //:requirements_3_13.update -- --upgrade
    ```

## Update Bazel Dependencies

1.  Create a local copy of the change by cloning the repository.

    ```shell
    $ git clone \
        https://github.com/google-parfait/federated-language/ \
        "/tmp/lang"
    $ cd "/tmp/lang"
    ```

1.  Update
    [MODULE.bazel](https://github.com/google-parfait/federated-language/blob/main/federated_language/MODULE.bazel)
    or
    [.bazelversion](https://github.com/google-parfait/federated-language/blob/main/federated_language/.bazelversion).

1.  Update the [Bazel lockfile](https://bazel.build/external/lockfile).

    ```shell
    $ bazelisk mod deps --lockfile_mode=update
    ```

## Debug Python Dependencies

```shell
$ python3 -m venv "venv"
$ source "venv/bin/activate"
$ pip install --upgrade pip
$ pip install --upgrade pip-tools
$ pip-compile --output-file=- "requirements.in" 1>/dev/null
```
