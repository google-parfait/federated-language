# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines classes/functions to manipulate the API context stack."""

from federated_language.context_stack import context


class RuntimeErrorContext(context.SyncContext):
  """A context that will fail if you execute against it."""

  def _raise_runtime_error(self):
    raise RuntimeError(
        'No default context installed.\n'
        '\n'
        'You should not expect to get this error using the TFF API.\n'
        '\n'
        'If you are getting this error when testing a module inside of '
        '`federated_language/python/core/...`, you may need to explicitly '
        'invoke `execution_contexts.set_sync_local_cpp_execution_context()` in '
        'the `main` function of your test.'
    )

  def invoke(self, comp, arg):
    del comp  # Unused
    del arg  # Unused
    self._raise_runtime_error()
