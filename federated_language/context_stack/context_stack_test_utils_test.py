# Copyright 2022 Google LLC
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

import asyncio
import contextlib
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.context_stack import context
from federated_language.context_stack import context_stack_impl
from federated_language.context_stack import context_stack_test_utils


class _TestContext(context.SyncContext):

  def invoke(self, comp, arg):
    raise AssertionError


class WithContextTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def test_installs_context_fn_sync_no_arg(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    @context_stack_test_utils.with_context(context_fn)
    def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, test_context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

  def test_installs_context_fn_sync_args(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    @context_stack_test_utils.with_context(context_fn)
    def _foo(x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, test_context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      _foo(1)

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

  def test_installs_context_fn_sync_kwargs(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    @context_stack_test_utils.with_context(context_fn)
    def _foo(*, x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, test_context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      _foo(x=1)

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

  def test_installs_context_fn_sync_return(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    @context_stack_test_utils.with_context(context_fn)
    def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, test_context)
      return 1

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      x = _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      # Assert that the return value is returned by the decorator.
      self.assertEqual(x, 1)

  async def test_installs_context_fn_async(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    @context_stack_test_utils.with_context(context_fn)
    async def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, test_context)

    # Assert that an async function is returned.
    self.assertTrue(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      await _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

  def test_installs_context_test_case(self):
    test_context = _TestContext()
    context_fn = lambda: test_context

    class _FooTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

      @context_stack_test_utils.with_context(context_fn)
      async def test_async(self):
        self.assertEqual(context_stack_impl.context_stack.current, test_context)

      @context_stack_test_utils.with_context(context_fn)
      def test_sync(self):
        self.assertEqual(context_stack_impl.context_stack.current, test_context)

      def test_undecorated(self):
        self.assertNotEqual(
            context_stack_impl.context_stack.current, test_context
        )

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_FooTest.test_sync))

    # Assert that an async function is returned.
    self.assertTrue(asyncio.iscoroutinefunction(_FooTest.test_async))

    with mock.patch.object(contextlib.ExitStack, 'enter_context'):
      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )

      # Assert that the test passes with the expected number of test cases.
      suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FooTest)
      self.assertEqual(suite.countTestCases(), 3)
      runner = unittest.TextTestRunner()
      result = runner.run(suite)
      self.assertEqual(result.testsRun, 3)
      self.assertTrue(result.wasSuccessful())

      # Assert that the context is not installed.
      self.assertNotEqual(
          context_stack_impl.context_stack.current, test_context
      )


if __name__ == '__main__':
  absltest.main()
