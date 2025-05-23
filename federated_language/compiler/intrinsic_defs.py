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
"""Definitions of all intrinsic for use within the system."""

import enum
from typing import Optional

from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_factory
import numpy as np

_intrinsic_registry = {}


@enum.unique
class BroadcastKind(enum.Enum):
  DEFAULT = 1
  SECURE = 2


@enum.unique
class AggregationKind(enum.Enum):
  DEFAULT = 1
  SECURE = 2


class IntrinsicDef:
  """Represents the definition of an intrinsic.

  This class represents the ultimate source of ground truth about what kinds of
  intrinsics exist and what their types are. To be consuled by all other code
  that deals with intrinsics.
  """

  def __init__(
      self,
      name: str,
      uri: str,
      type_signature: computation_types.Type,
      aggregation_kind: Optional[AggregationKind] = None,
      broadcast_kind: Optional[BroadcastKind] = None,
  ):
    """Constructs a definition of an intrinsic.

    Args:
      name: The short human-friendly name of this intrinsic.
      uri: The URI of this intrinsic.
      type_signature: The type of the intrinsic.
      aggregation_kind: Optional kind of aggregation performed by calls.
      broadcast_kind: Optional kind of broadcast performed by calls.
    """
    self._name = str(name)
    self._uri = str(uri)
    self._type_signature = type_signature
    self._aggregation_kind = aggregation_kind
    self._broadcast_kind = broadcast_kind
    _intrinsic_registry[str(uri)] = self

  @property
  def name(self):
    return self._name

  @property
  def uri(self):
    return self._uri

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def aggregation_kind(self) -> Optional[AggregationKind]:
    return self._aggregation_kind

  @property
  def broadcast_kind(self) -> Optional[BroadcastKind]:
    return self._broadcast_kind

  def __str__(self):
    return self._name

  def __repr__(self):
    return "IntrinsicDef('{}')".format(self._uri)


# Computes an aggregate of client items (the first, {T}@CLIENTS-typed parameter)
# using a multi-stage process in which client items are first partially
# aggregated at an intermediate layer, then the partial aggregates are further
# combined, and finally projected into the result. This multi-stage process is
# parameterized by a four-part aggregation interface that consists of the
# following:
# a) The 'zero' in the algebra used at the initial stage (partial aggregation),
#    This is the second, U-typed parameter.
# b) The operator that accumulates T-typed client items into the U-typed partial
#    aggregates. This is the third, (<U,T>->U)-typed parameter.
# c) The operator that combines pairs of U-typed partial aggregates. This is the
#    fourth, (<U,U>->U)-typed parameter.
# d) The operator that projects the top-level aggregate into the final result.
#    This is the fifth, (U->R)-typed parameter.
#
# Conceptually, given a new literal INTERMEDIATE_AGGREGATORS in a single-layer
# aggregation architecture, one could define this intrinsic in terms of generic
# intrinsics defined above, as follows.
#
# @federated_computation
# def federated_aggregate(x, zero, accumulate, merge, report):
#   a = generic_partial_reduce(x, zero, accumulate, INTERMEDIATE_AGGREGATORS)
#   b = generic_reduce(a, zero, merge, SERVER)
#   c = generic_map(report, b)
#   return c
#
# Actual implementations might vary.
#
# Type signature: <{T}@CLIENTS,U,(<U,T>->U),(<U,U>->U),(U->R)> -> R@SERVER
FEDERATED_AGGREGATE = IntrinsicDef(
    'FEDERATED_AGGREGATE',
    'federated_aggregate',
    computation_types.FunctionType(
        parameter=[
            computation_types.FederatedType(
                computation_types.AbstractType('T'), placements.CLIENTS
            ),
            computation_types.AbstractType('U'),
            type_factory.reduction_op(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('T'),
            ),
            type_factory.binary_op(computation_types.AbstractType('U')),
            computation_types.FunctionType(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('R'),
            ),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('R'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Applies a given function to a value on the server.
#
# Type signature: <(T->U),T@SERVER> -> U@SERVER
FEDERATED_APPLY = IntrinsicDef(
    'FEDERATED_APPLY',
    'federated_apply',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U'),
            ),
            computation_types.FederatedType(
                computation_types.AbstractType('T'), placements.SERVER
            ),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('U'), placements.SERVER
        ),
    ),
)

# Broadcasts a server item to all clients.
#
# Type signature: T@SERVER -> T@CLIENTS
FEDERATED_BROADCAST = IntrinsicDef(
    'FEDERATED_BROADCAST',
    'federated_broadcast',
    computation_types.FunctionType(
        parameter=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'),
            placements.CLIENTS,
            all_equal=True,
        ),
    ),
    broadcast_kind=BroadcastKind.DEFAULT,
)

# Evaluates a function at the clients.
#
# Type signature: (() -> T) -> {T}@CLIENTS
FEDERATED_EVAL_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_EVAL_AT_CLIENTS',
    'federated_eval_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.FunctionType(
            None, computation_types.AbstractType('T')
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS
        ),
    ),
)

# Evaluates a function at the server.
#
# Type signature: (() -> T) -> T@SERVER
FEDERATED_EVAL_AT_SERVER = IntrinsicDef(
    'FEDERATED_EVAL_AT_SERVER',
    'federated_eval_at_server',
    computation_types.FunctionType(
        parameter=computation_types.FunctionType(
            None, computation_types.AbstractType('T')
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
)

# Maps member constituents of a client value pointwise using a given mapping
# function that operates independently on each client.
#
# Type signature: <(T->U),{T}@CLIENTS> -> {U}@CLIENTS
FEDERATED_MAP = IntrinsicDef(
    'FEDERATED_MAP',
    'federated_map',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U'),
            ),
            computation_types.FederatedType(
                computation_types.AbstractType('T'), placements.CLIENTS
            ),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('U'), placements.CLIENTS
        ),
    ),
)

# Maps member constituents of a client all equal value pointwise using a given
# mapping function that operates independently on each client, as a result of
# this independence, the value is only garunteed to be all equal if the function
# is deterministic.
#
# Type signature: <(T->U),T@CLIENTS> -> U@CLIENTS
FEDERATED_MAP_ALL_EQUAL = IntrinsicDef(
    'FEDERATED_MAP_ALL_EQUAL',
    'federated_map_all_equal',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U'),
            ),
            computation_types.FederatedType(
                computation_types.AbstractType('T'),
                placements.CLIENTS,
                all_equal=True,
            ),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('U'),
            placements.CLIENTS,
            all_equal=True,
        ),
    ),
)

# Computes a simple (equally weighted) mean of client items. Only supported
# for numeric tensor types, or composite structures made up of numeric types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_MEAN = IntrinsicDef(
    'FEDERATED_MEAN',
    'federated_mean',
    computation_types.FunctionType(
        parameter=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Computes the min of client values on the server. Only supported for numeric
# types, or nested structures made up of numeric computation_types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_MIN = IntrinsicDef(
    'FEDERATED_MIN',
    'federated_min',
    computation_types.FunctionType(
        parameter=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Computes the max of client values on the server. Only supported for numeric
# types, or nested structures made up of numeric computation_types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_MAX = IntrinsicDef(
    'FEDERATED_MAX',
    'federated_max',
    computation_types.FunctionType(
        parameter=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Computes the sum of client values on the server, securely. Only supported for
# integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,M> -> V@SERVER
FEDERATED_SECURE_SUM = IntrinsicDef(
    'FEDERATED_SECURE_SUM',
    'federated_secure_sum',
    computation_types.FunctionType(
        parameter=[
            computation_types.FederatedType(
                computation_types.AbstractType('V'), placements.CLIENTS
            ),
            computation_types.AbstractType('M'),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('V'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.SECURE,
)

# Computes the sum of client values on the server, securely. Only supported for
# integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,B> -> V@SERVER
FEDERATED_SECURE_SUM_BITWIDTH = IntrinsicDef(
    'FEDERATED_SECURE_SUM_BITWIDTH',
    'federated_secure_sum_bitwidth',
    computation_types.FunctionType(
        parameter=[
            computation_types.FederatedType(
                computation_types.AbstractType('V'), placements.CLIENTS
            ),
            computation_types.AbstractType('B'),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('V'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.SECURE,
)

_SELECT_TYPE = computation_types.FunctionType(
    parameter=[
        computation_types.FederatedType(
            computation_types.AbstractType('Ks'), placements.CLIENTS
        ),  # client_keys
        computation_types.FederatedType(np.int32, placements.SERVER),  # max_key
        computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),  # server_state
        computation_types.FunctionType(
            [computation_types.AbstractType('T'), np.int32],
            computation_types.AbstractType('U'),
        ),  # select_fn
    ],
    result=computation_types.FederatedType(
        computation_types.SequenceType(computation_types.AbstractType('U')),
        placements.CLIENTS,
    ),
)

# Distributes server values to clients based on client keys.
FEDERATED_SELECT = IntrinsicDef(
    'FEDERATED_SELECT',
    'federated_select',
    _SELECT_TYPE,
    broadcast_kind=BroadcastKind.DEFAULT,
)

# Securely distributes server values to clients based on private client keys.
FEDERATED_SECURE_SELECT = IntrinsicDef(
    'FEDERATED_SECURE_SELECT',
    'federated_secure_select',
    _SELECT_TYPE,
    broadcast_kind=BroadcastKind.SECURE,
)

# Computes the sum of client values on the server. Only supported for numeric
# types, or nested structures made up of numeric computation_types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_SUM = IntrinsicDef(
    'FEDERATED_SUM',
    'federated_sum',
    computation_types.FunctionType(
        parameter=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS
        ),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Places a value at the clients.
#
# Type signature: T -> T@CLIENTS
FEDERATED_VALUE_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_VALUE_AT_CLIENTS',
    'federated_value_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.CLIENTS, True
        ),
    ),
)

# Places a value at the server.
#
# Type signature: T -> T@SERVER
FEDERATED_VALUE_AT_SERVER = IntrinsicDef(
    'FEDERATED_VALUE_AT_SERVER',
    'federated_value_at_server',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
)

# Computes a weighted mean of client items. Only supported for numeric tensor
# types, or composite structures made up of numeric types. Weights must be
# simple scalar numeric (integer or floating point, not complex) tensor types.
# The types of weights and values must be compatible, i.e., multiplying and
# dividing member constituents of the value by weights should yield results of
# the same type as the type of these member constituents being weighted. Thus,
# for example, one may not supply values containing `np.int32`` tensors, as the
# result of weighting such values is of a floating-point type. Casting must be
# injected, where appropriate, by the compiler before invoking this intrinsic.
#
# Type signature: <{T}@CLIENTS,{U}@CLIENTS> -> T@SERVER
FEDERATED_WEIGHTED_MEAN = IntrinsicDef(
    'FEDERATED_WEIGHTED_MEAN',
    'federated_weighted_mean',
    computation_types.FunctionType(
        parameter=[
            computation_types.FederatedType(
                computation_types.AbstractType('T'), placements.CLIENTS
            ),
            computation_types.FederatedType(
                computation_types.AbstractType('U'), placements.CLIENTS
            ),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('T'), placements.SERVER
        ),
    ),
    aggregation_kind=AggregationKind.DEFAULT,
)

# Zips a tuple of two federated types into a federated tuple.
#
# Type signature: T -> U@CLIENTS
# where `T` is a structure of client-placed values.
# `U` must be identical to `T` with federated placement removed.
FEDERATED_ZIP_AT_CLIENTS = IntrinsicDef(
    'FEDERATED_ZIP_AT_CLIENTS',
    'federated_zip_at_clients',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.FederatedType(
            computation_types.AbstractType('U'), placements.CLIENTS
        ),
    ),
)
# Type signature: T -> U@SERVER
# where `T` is a structure of server-placed values.
# `U` must be identical to `T` with federated placement removed.
FEDERATED_ZIP_AT_SERVER = IntrinsicDef(
    'FEDERATED_ZIP_AT_SERVER',
    'federated_zip_at_server',
    computation_types.FunctionType(
        parameter=computation_types.AbstractType('T'),
        result=computation_types.FederatedType(
            computation_types.AbstractType('U'), placements.SERVER
        ),
    ),
)

# Generic plus operator that accepts a variety of types in federated computation
# context. The range of types 'T' supported to be defined. It should work in a
# natural manner for tensors, tuples, federated types, possibly sequences, and
# maybe even functions (although it's unclear if such generality is desirable).
#
# Type signature: <T,T> -> T
GENERIC_PLUS = IntrinsicDef(
    'GENERIC_PLUS',
    'generic_plus',
    type_factory.binary_op(computation_types.AbstractType('T')),
)

# Performs pointwise TensorFlow division on its arguments.
# The type signature of generic divide is determined by TensorFlow's set of
# implicit type equations. For example, dividing `int32` by `int32` in TF
# generates a tensor of type `float64`. There is therefore more structure than
# is suggested by the type signature `<T,T> -> U`.
#
# Type signature: <T,T> -> U
GENERIC_DIVIDE = IntrinsicDef(
    'GENERIC_DIVIDE',
    'generic_divide',
    computation_types.FunctionType(
        [
            computation_types.AbstractType('T'),
            computation_types.AbstractType('T'),
        ],
        computation_types.AbstractType('U'),
    ),
)

# Performs pointwise TensorFlow multiplication on its arguments.
#
# Type signature: <T,T> -> T
GENERIC_MULTIPLY = IntrinsicDef(
    'GENERIC_MULTIPLY',
    'generic_multiply',
    computation_types.FunctionType(
        [computation_types.AbstractType('T')] * 2,
        computation_types.AbstractType('T'),
    ),
)
# Generic zero operator that represents zero-filled values of diverse types (to
# be defined, but generally similar to that supported by GENERIC_ADD).
#
# Type signature: T
GENERIC_ZERO = IntrinsicDef(
    'GENERIC_ZERO', 'generic_zero', computation_types.AbstractType('T')
)

# Maps elements of a sequence using a given mapping function that operates
# independently on each element.
#
# Type signature: <(T->U),T*> -> U*
SEQUENCE_MAP = IntrinsicDef(
    'SEQUENCE_MAP',
    'sequence_map',
    computation_types.FunctionType(
        parameter=[
            computation_types.FunctionType(
                computation_types.AbstractType('T'),
                computation_types.AbstractType('U'),
            ),
            computation_types.SequenceType(computation_types.AbstractType('T')),
        ],
        result=computation_types.SequenceType(
            computation_types.AbstractType('U')
        ),
    ),
)

# Reduces a sequence using a given 'zero' in the algebra (i.e., the result of
# reducing an empty sequence) and a given reduction operator with the signature
# U,T->U that incorporates a single T-typed element into a U-typed result of
# partial reduction. In the special case of T = U, this corresponds to the
# classical notion of reduction of a set using a commutative associative binary
# operator. The generalized reduction operator (with T != U) must yield the same
# results when repeatedly applied on sequences of elements in any order.
#
# Type signature: <T*,U,(<U,T>->U)> -> U
SEQUENCE_REDUCE = IntrinsicDef(
    'SEQUENCE_REDUCE',
    'sequence_reduce',
    computation_types.FunctionType(
        parameter=[
            computation_types.SequenceType(computation_types.AbstractType('T')),
            computation_types.AbstractType('U'),
            type_factory.reduction_op(
                computation_types.AbstractType('U'),
                computation_types.AbstractType('T'),
            ),
        ],
        result=computation_types.AbstractType('U'),
    ),
)

# Computes the sum of values in a sequence. Only supported for numeric types
# or nested structures made up of numeric types.
#
# Type signature: T* -> T
SEQUENCE_SUM = IntrinsicDef(
    'SEQUENCE_SUM',
    'sequence_sum',
    computation_types.FunctionType(
        parameter=computation_types.SequenceType(
            computation_types.AbstractType('T')
        ),
        result=computation_types.AbstractType('T'),
    ),
)


def uri_to_intrinsic_def(uri) -> Optional[IntrinsicDef]:
  return _intrinsic_registry.get(uri)


# TODO: b/254770431 - Add documentation explaining the implications of setting
# broadcast_kind for an intrinsic.
def get_broadcast_intrinsics() -> list[IntrinsicDef]:
  return [
      intrinsic
      for intrinsic in _intrinsic_registry.values()
      if intrinsic.broadcast_kind
  ]


# TODO: b/254770431 - Add documentation explaining the implications of setting
# aggregation_kind for an intrinsic.
def get_aggregation_intrinsics() -> list[IntrinsicDef]:
  return [
      intrinsic
      for intrinsic in _intrinsic_registry.values()
      if intrinsic.aggregation_kind
  ]
