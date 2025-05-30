# Copyright 2019 Google LLC
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
# See the License for the specific language governing permisions and
# limitations under the License.
"""A library of static analysis functions for ASTs."""

from collections.abc import Callable, Iterable
from typing import Optional, Union

from federated_language.compiler import building_block_analysis
from federated_language.compiler import building_blocks
from federated_language.compiler import intrinsic_defs
from federated_language.compiler import transformation_utils
from federated_language.types import computation_types
from federated_language.types import placements
from federated_language.types import type_analysis


_TypeOrTupleOfTypes = Union[
    type[building_blocks.ComputationBuildingBlock],
    tuple[type[building_blocks.ComputationBuildingBlock], ...],
]


def visit_preorder(
    tree: building_blocks.ComputationBuildingBlock,
    function: Callable[[building_blocks.ComputationBuildingBlock], None],
):

  def _visit(building_block):
    function(building_block)
    return building_block, False

  transformation_utils.transform_preorder(tree, _visit)


def visit_postorder(
    tree: building_blocks.ComputationBuildingBlock,
    function: Callable[[building_blocks.ComputationBuildingBlock], None],
):

  def _visit(building_block):
    function(building_block)
    return building_block, False

  transformation_utils.transform_postorder(tree, _visit)


_BuildingBlockPredicate = Callable[
    [building_blocks.ComputationBuildingBlock], bool
]


def count(
    tree: building_blocks.ComputationBuildingBlock,
    predicate: Optional[_BuildingBlockPredicate] = None,
) -> int:
  """Returns the number of building blocks in `tree` matching `predicate`.

  Args:
    tree: A tree of `building_blocks.ComputationBuildingBlock`s to count.
    predicate: An optional Python function that takes a tree as a parameter and
      returns a boolean value. If `None`, all computations are counted.
  """
  counter = 0

  def _fn(building_block):
    nonlocal counter
    if predicate is None or predicate(building_block):
      counter += 1

  visit_postorder(tree, _fn)
  return counter


def contains(
    tree: building_blocks.ComputationBuildingBlock,
    predicate: _BuildingBlockPredicate,
) -> bool:
  """Returns whether or not a building block in `tree` matches `predicate`."""
  return count(tree, predicate) != 0


def contains_only(
    tree: building_blocks.ComputationBuildingBlock,
    predicate: _BuildingBlockPredicate,
) -> bool:
  """Returns whether or not a building block in `tree` matches `predicate`."""
  return not contains(tree, lambda x: not predicate(x))


def check_has_single_placement(
    comp: building_blocks.ComputationBuildingBlock,
    single_placement: placements.PlacementLiteral,
) -> None:
  """Checks that the AST of `comp` contains only `single_placement`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock`.
    single_placement: Instance of `placements.PlacementLiteral` which should be
      the only placement present under `comp`.

  Raises:
    ValueError: If the AST under `comp` contains any
    `computation_types.FederatedType` other than `single_placement`.
  """

  def _check_single_placement(comp):
    """Checks that the placement in `type_spec` matches `single_placement`."""
    if (
        isinstance(comp.type_signature, computation_types.FederatedType)
        and comp.type_signature.placement is not single_placement
    ):
      raise ValueError(
          'Comp contains a placement other than {}; '
          'placement {} on comp {} inside the structure. '.format(
              single_placement,
              comp.type_signature.placement,
              comp.compact_representation(),
          )
      )

  visit_postorder(comp, _check_single_placement)


def check_contains_only_reducible_intrinsics(
    comp: building_blocks.ComputationBuildingBlock,
):
  """Checks that `comp` contains intrinsics reducible to aggregate or broadcast.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to check for
      presence of intrinsics not currently immediately reducible to
      `FEDERATED_AGGREGATE` or `FEDERATED_BROADCAST`, or local processing.

  Raises:
    ValueError: If we encounter an intrinsic under `comp` that is not reducible.
  """
  reducible_uris = (
      intrinsic_defs.FEDERATED_AGGREGATE.uri,
      intrinsic_defs.FEDERATED_APPLY.uri,
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri,
      intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_MAP.uri,
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
      intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH.uri,
      intrinsic_defs.FEDERATED_SECURE_SUM.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
  )

  def _check(comp):
    if (
        isinstance(comp, building_blocks.Intrinsic)
        and comp.uri not in reducible_uris
    ):
      raise ValueError(
          'Encountered an Intrinsic not currently reducible to aggregate or '
          'broadcast, the intrinsic {}'.format(comp.compact_representation())
      )

  visit_postorder(comp, _check)


class NonuniqueNameError(ValueError):

  def __init__(self, comp, name):
    self.comp = comp
    self.name = name
    message = (
        f'The name `{name}` is bound multiple times in the computation:\n'
        f'{comp.compact_representation()}'
    )
    super().__init__(message)


def check_has_unique_names(
    comp: building_blocks.ComputationBuildingBlock,
) -> None:
  """Checks that each variable of `comp` is bound at most once.

  Additionally, checks that `comp` does not mask any names which are unbound
  at the top level.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock`.

  Raises:
    NonuniqueNameError: If we encounter a name that is bound multiple times or a
      binding which would shadow an unbound reference.
  """
  # Initializing `names` to unbound names in `comp` ensures that `comp` does not
  # mask any names from its parent scope.
  names = transformation_utils.get_map_of_unbound_references(comp)[comp]

  def _visit_name(name):
    if name in names:
      raise NonuniqueNameError(comp, name)
    names.add(name)

  def _visit(comp):
    if isinstance(comp, building_blocks.Block):
      for name, _ in comp.locals:
        _visit_name(name)
    elif (
        isinstance(comp, building_blocks.Lambda)
        and comp.parameter_type is not None
    ):
      _visit_name(comp.parameter_name)

  visit_postorder(comp, _visit)


def _extract_nodes_consuming(
    tree: building_blocks.ComputationBuildingBlock,
    predicate: _BuildingBlockPredicate,
) -> set[building_blocks.ComputationBuildingBlock]:
  """Returns the set of AST nodes which consume nodes matching `predicate`.

  Notice we adopt the convention that a node which itself satisfies the
  predicate is in this set.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to view as an
      abstract syntax tree, and construct the set of nodes in this tree having a
      dependency on nodes matching `predicate`; that is, the set of nodes whose
      value depends on evaluating nodes matching `predicate`.
    predicate: One-arg callable, accepting arguments of type
      `building_blocks.ComputationBuildingBlock` and returning a `bool`
      indicating match or mismatch with the desired pattern.

  Returns:
    A `set` of `building_blocks.ComputationBuildingBlock` instances
    representing the nodes in `tree` dependent on nodes matching `predicate`.
  """

  class _NodeSet:

    def __init__(self):
      self.mapping = {}

    def add(self, comp):
      self.mapping[id(comp)] = comp

    def to_set(self):
      return set(self.mapping.values())

  dependent_nodes = _NodeSet()

  def _are_children_in_dependent_set(comp, symbol_tree):
    """Checks if the dependencies of `comp` are present in `dependent_nodes`."""
    if isinstance(
        comp,
        (
            building_blocks.CompiledComputation,
            building_blocks.Data,
            building_blocks.Intrinsic,
            building_blocks.Literal,
            building_blocks.Placement,
        ),
    ):
      return False
    elif isinstance(comp, building_blocks.Lambda):
      return id(comp.result) in dependent_nodes.mapping
    elif isinstance(comp, building_blocks.Block):
      return (
          any(id(x[1]) in dependent_nodes.mapping for x in comp.locals)
          or id(comp.result) in dependent_nodes.mapping
      )
    elif isinstance(comp, building_blocks.Struct):
      return any(id(x) in dependent_nodes.mapping for x in comp)
    elif isinstance(comp, building_blocks.Selection):
      return id(comp.source) in dependent_nodes.mapping
    elif isinstance(comp, building_blocks.Call):
      return (
          id(comp.function) in dependent_nodes.mapping
          or id(comp.argument) in dependent_nodes.mapping
      )
    elif isinstance(comp, building_blocks.Reference):
      return _is_reference_dependent(comp, symbol_tree)

  def _is_reference_dependent(comp, symbol_tree):
    payload = symbol_tree.get_payload_with_name(comp.name)
    if payload is None:
      return False
    # The postorder traversal ensures that we process any
    # bindings before we process the reference to those bindings
    return id(payload.value) in dependent_nodes.mapping

  def _populate_dependent_set(comp, symbol_tree):
    """Populates `dependent_nodes` with all nodes dependent on `predicate`."""
    if predicate(comp):
      dependent_nodes.add(comp)
    elif _are_children_in_dependent_set(comp, symbol_tree):
      dependent_nodes.add(comp)
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter
  )
  transformation_utils.transform_postorder_with_symbol_bindings(
      tree, _populate_dependent_set, symbol_tree
  )
  return dependent_nodes.to_set()


def _extract_calls_with_fn_consuming_arg(
    tree: building_blocks.ComputationBuildingBlock,
    *,
    fn_predicate: _BuildingBlockPredicate,
    arg_predicate: _BuildingBlockPredicate,
) -> list[building_blocks.Call]:
  """Extracts calls depending on function and arg predicates.

  This function returns all calls in `tree` whose fns consume nodes matching
  `fn_predicate` and arguments consume nodes matching `arg_predicate`. This
  matching can be useful in checking that one type of function does not consume
  any nodes depending on another type of function in the body of `tree`.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to traverse.
    fn_predicate: Callable taking a building block and returning a boolean, to
      define the behavior of this function according to the semantics above.
    arg_predicate: Callable taking a building block and returning a boolean, to
      define the behavior of this function according to the semantics above.

  Returns:
    A list of `building_block.Calls` matching the description above.
  """
  nodes_dependent_on_arg_predicate = _extract_nodes_consuming(
      tree, arg_predicate
  )

  nodes_dependent_on_fn_predicate = _extract_nodes_consuming(tree, fn_predicate)

  instances = []

  for node in nodes_dependent_on_arg_predicate:
    if isinstance(node, building_blocks.Call):
      if (
          node.argument in nodes_dependent_on_arg_predicate
          and node.function in nodes_dependent_on_fn_predicate
      ):
        instances.append(node)
  return instances


def check_broadcast_not_dependent_on_aggregate(
    tree: building_blocks.ComputationBuildingBlock,
) -> None:
  """Raises if any broadcast in `tree` ingests the result of an aggregate.

  We explicitly check for this pattern since if it occurs, `tree` is not
  reducible to broadcast-map-aggregate form.


  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to check for
      the presence of a broadcast which ingests the result of an aggregate.

  Raises:
    ValueError: If a broadcast in `tree` consumes the result of an aggregate.
  """

  def aggregate_predicate(x):
    return (
        isinstance(x, building_blocks.Intrinsic)
        and x.intrinsic_def().aggregation_kind
    )

  def broadcast_predicate(x):
    return (
        isinstance(x, building_blocks.Intrinsic)
        and x.intrinsic_def().broadcast_kind
    )

  broadcast_dependent_examples = _extract_calls_with_fn_consuming_arg(
      tree, fn_predicate=broadcast_predicate, arg_predicate=aggregate_predicate
  )
  if broadcast_dependent_examples:
    raise ValueError(
        'Detected broadcast dependent on aggregate. Examples are: {}'.format(
            broadcast_dependent_examples
        )
    )


def check_aggregate_not_dependent_on_aggregate(
    tree: building_blocks.ComputationBuildingBlock,
) -> None:
  """Raises if any aggregation in `tree` ingests the result of an aggregate.

  We explicitly check for this pattern since if it occurs, `tree` is not
  reducible to `MergeableCompForm`.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to check for
      the presence of an aggregation which ingests the result of another
      aggregate.

  Raises:
    ValueError: If a broadcast in `tree` consumes the result of an aggregate.
  """

  def aggregate_predicate(x):
    return (
        isinstance(x, building_blocks.Intrinsic)
        and x.intrinsic_def().aggregation_kind
    )

  multiple_agg_dependent_examples = _extract_calls_with_fn_consuming_arg(
      tree, fn_predicate=aggregate_predicate, arg_predicate=aggregate_predicate
  )
  if multiple_agg_dependent_examples:
    raise ValueError(
        'Detected one aggregate dependent on another. Examples are: {}'.format(
            multiple_agg_dependent_examples
        )
    )


def check_contains_no_unbound_references(tree, excluding=None):
  """Checks that `tree` has no unbound references.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to view as an
      abstract syntax tree.
    excluding: A `string` or a collection of `string`s representing the names of
      references to exclude from the test.

  Raises:
    ValueError: If `comp` has unbound references.
  """
  if not contains_no_unbound_references(tree, excluding):
    raise ValueError(
        'The AST contains unbound references: {}.'.format(
            tree.formatted_representation()
        )
    )


def check_contains_no_new_unbound_references(old_tree, new_tree):
  """Checks that `new_tree` contains no unbound references not in `old_tree`."""
  old_unbound = transformation_utils.get_map_of_unbound_references(old_tree)[
      old_tree
  ]
  new_unbound = transformation_utils.get_map_of_unbound_references(new_tree)[
      new_tree
  ]
  diff = new_unbound - old_unbound
  if diff:
    raise ValueError(
        'Expected no new unbounded references. '
        f'Old tree:\n{old_tree}\nNew tree:\n{new_tree}\n'
        f'New unbound references: {diff}'
    )


def contains_called_intrinsic(tree, uri=None):
  """Tests if `tree` contains a called intrinsic for the given `uri`.

  Args:
    tree: A `building_blocks.ComputationBuildingBlock`.
    uri: An optional URI or list of URIs; the same as what is accepted by
      `building_block_analysis.is_called_intrinsic`.

  Returns:
    `True` if there is a called intrinsic in `tree` for the given `uri`,
    otherwise `False`.
  """
  predicate = lambda x: building_block_analysis.is_called_intrinsic(x, uri)
  return count(tree, predicate) > 0


def contains_no_unbound_references(
    tree: building_blocks.ComputationBuildingBlock,
    excluding: Optional[Union[str, Iterable[str]]] = None,
) -> bool:
  """Tests if all the references in `tree` are bound by `tree`.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to view as an
      abstract syntax tree.
    excluding: A `string` or a collection of `string`s representing the names of
      references to exclude from the test.

  Returns:
    `True` if there are no unbound references in `tree` excluding those
    specified by `excluding`, otherwise `False`.
  """
  if isinstance(excluding, str):
    excluding = [excluding]
  unbound_references = transformation_utils.get_map_of_unbound_references(tree)
  if excluding is not None:
    excluding = set(excluding)
    names = unbound_references[tree] - excluding
  else:
    names = unbound_references[tree]
  num_unbound_references = len(names)
  return num_unbound_references == 0


_DEFAULT_KIND_PREDICATE = lambda k: k is not None


def find_aggregations_in_tree(
    comp: building_blocks.ComputationBuildingBlock,
    kind_predicate: Callable[
        [intrinsic_defs.AggregationKind], bool
    ] = _DEFAULT_KIND_PREDICATE,
) -> list[building_blocks.Call]:
  """Finds aggregating calls with kind matching `kind_predicate` in `comp`.

  An "aggregating call" for the purpose of this function is a call to an
  intrinsic which takes values at CLIENT and materializes some result at
  SERVER.

  Args:
    comp: An AST to search.
    kind_predicate: A filter for kind of aggregation to search for.

  Returns:
    A list of child ASTs which are calls to aggregating intrinsics with kinds
    matching `aggregation_kind`.

  Raises:
    ValueError if `comp` contains a call whose target function cannot be
      identified. This may result from calls to references or other
      indirect structures.
  """
  aggregation_calls: list[building_blocks.Call] = []

  def record_intrinsic_calls(comp):
    """Identifies matching calls and adds them to `aggregation_calls`."""
    if not isinstance(comp, building_blocks.Call):
      return
    # Called lambdas will only trigger aggregation if they themselves contain
    # aggregation, which will be caught when the lambea itself is traversed.
    if isinstance(comp.function, building_blocks.Lambda):
      return
    # Aggregation cannot be occurring if the output type is not federated
    if not isinstance(
        comp.function.type_signature, computation_types.FunctionType
    ):
      return
    if not type_analysis.contains_federated_types(
        comp.function.type_signature.result
    ):
      return

    # We can't tell whether an arbitrary AST fragment results in an intrinsic
    # with a given URI, so we report an error in this case.
    if not isinstance(comp.function, building_blocks.Intrinsic):
      raise ValueError(
          'Cannot determine whether call contains aggregation: ' + str(comp)
      )

    # Aggregation with inputs that don't contain any tensors isn't interesting.
    #
    # NOTE: this is only applicable to intrinsic calls. Users can write their
    # own functions that internally materialize values at clients + aggregate
    # without taking any input tensors.
    #
    # This means that this check *must* come after the check above ensuring
    # that we're only talking about calls to `building_blocks.Intrinsic`s.
    if comp.argument is None or not type_analysis.contains_tensor_types(
        comp.argument.type_signature
    ):
      return

    if kind_predicate(comp.function.intrinsic_def().aggregation_kind):
      aggregation_calls.append(comp)

  visit_postorder(comp, record_intrinsic_calls)
  return aggregation_calls
