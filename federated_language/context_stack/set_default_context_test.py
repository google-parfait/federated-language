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

from absl.testing import absltest
from federated_language.context_stack import context_base
from federated_language.context_stack import context_stack_impl
from federated_language.context_stack import set_default_context


class _TestContext(context_base.SyncContext):
  """A test context."""

  def invoke(self, comp, arg):
    raise AssertionError


class SetDefaultContextTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # In these tests we are setting the default context of the
    # `context_stack_impl.context_stack`, so here we reset that context back to
    # some known state.
    self.context = _TestContext()
    context_stack_impl.context_stack.set_default_context(self.context)

  def test_with_context(self):
    context = _TestContext()
    context_stack = context_stack_impl.context_stack
    self.assertIsNot(context_stack.current, context)

    set_default_context.set_default_context(context)

    self.assertIs(context_stack.current, context)

  def test_raises_type_error_with_none(self):
    with self.assertRaises(TypeError):
      set_default_context.set_default_context(None)


if __name__ == '__main__':
  absltest.main()
