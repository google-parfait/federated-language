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
from federated_language.context_stack import context
from federated_language.context_stack import context_stack_impl


class _TestContext(context.SyncContext):

  def invoke(self, comp, arg):
    raise AssertionError


class ContextStackTest(absltest.TestCase):

  def test_set_default_context_with_context(self):
    default_context = _TestContext()
    context_stack = context_stack_impl.ContextStack(default_context)
    test_context = _TestContext()
    self.assertIsNot(context_stack.current, test_context)

    context_stack.set_default_context(test_context)

    self.assertIs(context_stack.current, test_context)

  def test_install_pushes_context_on_stack(self):
    default_context = _TestContext()
    context_stack = context_stack_impl.ContextStack(default_context)
    self.assertIs(context_stack.current, default_context)

    context_two = _TestContext()
    with context_stack.install(context_two):
      self.assertIs(context_stack.current, context_two)

      context_three = _TestContext()
      with context_stack.install(context_three):
        self.assertIs(context_stack.current, context_three)

      self.assertIs(context_stack.current, context_two)

    self.assertIs(context_stack.current, default_context)


class SetDefaultContextTest(absltest.TestCase):

  def test_with_context(self):
    test_context = _TestContext()
    context_stack = context_stack_impl.context_stack
    self.assertIsNot(context_stack.current, test_context)

    context_stack_impl.set_default_context(test_context)

    self.assertIs(context_stack.current, test_context)


if __name__ == '__main__':
  absltest.main()
