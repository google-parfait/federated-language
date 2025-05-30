# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.compiler import intrinsic_defs


def _get_intrinsic_named_parameters():
  def _predicate(obj):
    return isinstance(obj, intrinsic_defs.IntrinsicDef)

  objects = [getattr(intrinsic_defs, x) for x in dir(intrinsic_defs)]
  intrinsics = filter(_predicate, objects)
  return [(x.name, x) for x in intrinsics]


class IntrinsicDefsTest(parameterized.TestCase):

  @parameterized.named_parameters(*_get_intrinsic_named_parameters())
  def test_names_match_those_in_module(self, intrinsic):
    self.assertEqual(intrinsic, getattr(intrinsic_defs, intrinsic.name))

  def test_uris_are_unique(self):
    uris = set([x.uri for _, x in _get_intrinsic_named_parameters()])
    expected_length = len(_get_intrinsic_named_parameters())
    self.assertLen(uris, expected_length)

  @parameterized.named_parameters(
      ('federated_broadcast', 'FEDERATED_BROADCAST', '(T@SERVER -> T@CLIENTS)'),
      (
          'federated_eval_at_clients',
          'FEDERATED_EVAL_AT_CLIENTS',
          '(( -> T) -> {T}@CLIENTS)',
      ),
      (
          'federated_eval_at_server',
          'FEDERATED_EVAL_AT_SERVER',
          '(( -> T) -> T@SERVER)',
      ),
      (
          'federated_map',
          'FEDERATED_MAP',
          '(<(T -> U),{T}@CLIENTS> -> {U}@CLIENTS)',
      ),
      (
          'federated_secure_sum',
          'FEDERATED_SECURE_SUM',
          '(<{V}@CLIENTS,M> -> V@SERVER)',
      ),
      (
          'federated_secure_sum_bitwidth',
          'FEDERATED_SECURE_SUM_BITWIDTH',
          '(<{V}@CLIENTS,B> -> V@SERVER)',
      ),
      (
          'federated_secure_select',
          'FEDERATED_SECURE_SELECT',
          (
              '(<{Ks}@CLIENTS,int32@SERVER,T@SERVER,(<T,int32> -> U)> ->'
              ' {U*}@CLIENTS)'
          ),
      ),
      (
          'federated_select',
          'FEDERATED_SELECT',
          (
              '(<{Ks}@CLIENTS,int32@SERVER,T@SERVER,(<T,int32> -> U)> ->'
              ' {U*}@CLIENTS)'
          ),
      ),
      ('federated_sum', 'FEDERATED_SUM', '({T}@CLIENTS -> T@SERVER)'),
      (
          'federated_zip_at_clients',
          'FEDERATED_ZIP_AT_CLIENTS',
          '(T -> {U}@CLIENTS)',
      ),
      ('federated_zip_at_server', 'FEDERATED_ZIP_AT_SERVER', '(T -> U@SERVER)'),
  )
  def test_type_signature_strings(self, name, type_str):
    intrinsic = getattr(intrinsic_defs, name)
    self.assertEqual(
        intrinsic.type_signature.compact_representation(), type_str
    )


if __name__ == '__main__':
  absltest.main()
