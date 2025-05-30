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
"""A library of transformation utilities."""

import abc
import collections
from collections.abc import Callable
import itertools
import operator
import typing
from typing import Optional

from federated_language.compiler import building_blocks


def transform_postorder(
    comp: building_blocks.ComputationBuildingBlock, transform
):
  """Traverses `comp` recursively postorder and replaces its constituents.

  For each element of `comp` viewed as an expression tree, the transformation
  `transform` is applied first to building blocks it is parameterized by, then
  the element itself. The transformation `transform` should act as an identity
  function on the kinds of elements (computation building blocks) it does not
  care to transform. This corresponds to a post-order traversal of the
  expression tree, i.e., parameters are always transformed left-to-right (in
  the order in which they are listed in building block constructors), then the
  parent is visited and transformed with the already-visited, and possibly
  transformed arguments in place.

  NOTE: In particular, in `Call(f,x)`, both `f` and `x` are arguments to `Call`.
  Therefore, `f` is transformed into `f'`, next `x` into `x'` and finally,
  `Call(f',x')` is transformed at the end.

  Args:
    comp: A `computation_building_block.ComputationBuildingBlock` to traverse
      and transform bottom-up.
    transform: The transformation to apply locally to each building block in
      `comp`. It is a Python function that accepts a building block at input,
      and should return a (building block, bool) tuple as output, where the
      building block is a `computation_building_block.ComputationBuildingBlock`
      representing either the original building block or a transformed building
      block and the bool is a flag indicating if the building block was modified
      as.

  Returns:
    The result of applying `transform` to parts of `comp` in a bottom-up
    fashion, along with a Boolean with the value `True` if `comp` was
    transformed and `False` if it was not.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    NotImplementedError: If the argument is a kind of computation building block
      that is currently not recognized.
  """
  if isinstance(
      comp,
      (
          building_blocks.CompiledComputation,
          building_blocks.Data,
          building_blocks.Intrinsic,
          building_blocks.Literal,
          building_blocks.Placement,
          building_blocks.Reference,
      ),
  ):
    return transform(comp)
  elif isinstance(comp, building_blocks.Selection):
    source, source_modified = transform_postorder(comp.source, transform)
    if source_modified:
      comp = building_blocks.Selection(source, comp.name, comp.index)
    comp, comp_modified = transform(comp)
    return comp, comp_modified or source_modified
  elif isinstance(comp, building_blocks.Struct):
    elements = []
    elements_modified = False
    for key, value in comp.items():
      value, value_modified = transform_postorder(value, transform)
      elements.append((key, value))
      elements_modified = elements_modified or value_modified
    if elements_modified:
      comp = building_blocks.Struct(
          elements, container_type=comp.type_signature.python_container
      )
    comp, comp_modified = transform(comp)
    return comp, comp_modified or elements_modified
  elif isinstance(comp, building_blocks.Call):
    fn, fn_modified = transform_postorder(comp.function, transform)
    if comp.argument is not None:
      arg, arg_modified = transform_postorder(comp.argument, transform)
    else:
      arg, arg_modified = (None, False)
    if fn_modified or arg_modified:
      comp = building_blocks.Call(fn, arg)
    comp, comp_modified = transform(comp)
    return comp, comp_modified or fn_modified or arg_modified
  elif isinstance(comp, building_blocks.Lambda):
    result, result_modified = transform_postorder(comp.result, transform)
    if result_modified:
      comp = building_blocks.Lambda(
          comp.parameter_name, comp.parameter_type, result
      )
    comp, comp_modified = transform(comp)
    return comp, comp_modified or result_modified
  elif isinstance(comp, building_blocks.Block):
    variables = []
    variables_modified = False
    for key, value in comp.locals:
      value, value_modified = transform_postorder(value, transform)
      variables.append((key, value))
      variables_modified = variables_modified or value_modified
    result, result_modified = transform_postorder(comp.result, transform)
    if variables_modified or result_modified:
      comp = building_blocks.Block(variables, result)
    comp, comp_modified = transform(comp)
    return comp, comp_modified or variables_modified or result_modified
  else:
    raise NotImplementedError(f'Unexpected building block found: {type(comp)}.')


TransformReturnType = tuple[building_blocks.ComputationBuildingBlock, bool]


def transform_preorder(
    comp: building_blocks.ComputationBuildingBlock,
    transform: Callable[
        [building_blocks.ComputationBuildingBlock], TransformReturnType
    ],
) -> TransformReturnType:
  """Walks the AST of `comp` preorder, calling `transform` on the way down.

  Notice that this function will stop walking the tree when its transform
  function modifies a node; this is to prevent the caller from unexpectedly
  kicking off an infinite recursion. For this purpose the transform function
  must identify when it has transformed the structure of a building block; if
  the structure of the building block is modified but `False` is returned as
  the second element of the tuple returned by `transform`, `transform_preorder`
  may result in an infinite recursion.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to be
      transformed in a preorder fashion.
    transform: Transform function to be applied to the nodes of `comp`. Must
      return a two-tuple whose first element is a
      `building_blocks.ComputationBuildingBlock` and whose second element is a
      Boolean. If the computation which is passed to `comp` is returned in a
      modified state, must return `True` for the second element. This Boolean
      controls whether or not to stop traversing the tree under `comp`; if this
      Bool is `True`, `transform_preorder` will not traverse this subtree.

  Returns:
    A two-tuple, whose first element is modified version of `comp`, and
    whose second element is a Boolean indicating whether `comp` was transformed
    during the walk.
  """

  inner_comp, modified = transform(comp)
  if modified:
    return inner_comp, modified
  if isinstance(
      inner_comp,
      (
          building_blocks.CompiledComputation,
          building_blocks.Data,
          building_blocks.Intrinsic,
          building_blocks.Literal,
          building_blocks.Placement,
          building_blocks.Reference,
      ),
  ):
    return inner_comp, modified
  elif isinstance(inner_comp, building_blocks.Lambda):
    transformed_result, result_modified = transform_preorder(
        inner_comp.result, transform
    )
    if not (modified or result_modified):
      return inner_comp, False
    return (
        building_blocks.Lambda(
            inner_comp.parameter_name,
            inner_comp.parameter_type,
            transformed_result,
        ),
        True,
    )
  elif isinstance(inner_comp, building_blocks.Struct):
    elements_modified = False
    elements = []
    for name, val in inner_comp.items():
      result, result_modified = transform_preorder(val, transform)
      elements_modified = elements_modified or result_modified
      elements.append((name, result))
    if not (modified or elements_modified):
      return inner_comp, False
    return building_blocks.Struct(elements), True
  elif isinstance(inner_comp, building_blocks.Selection):
    transformed_source, source_modified = transform_preorder(
        inner_comp.source, transform
    )
    if not (modified or source_modified):
      return inner_comp, False
    return (
        building_blocks.Selection(
            transformed_source, inner_comp.name, inner_comp.index
        ),
        True,
    )
  elif isinstance(inner_comp, building_blocks.Call):
    transformed_fn, fn_modified = transform_preorder(
        inner_comp.function, transform
    )
    if inner_comp.argument is not None:
      transformed_arg, arg_modified = transform_preorder(
          inner_comp.argument, transform
      )
    else:
      transformed_arg = None
      arg_modified = False
    if not (modified or fn_modified or arg_modified):
      return inner_comp, False
    return building_blocks.Call(transformed_fn, transformed_arg), True
  elif isinstance(inner_comp, building_blocks.Block):
    transformed_variables = []
    values_modified = False
    for key, value in inner_comp.locals:
      transformed_value, value_modified = transform_preorder(value, transform)
      transformed_variables.append((key, transformed_value))
      values_modified = values_modified or value_modified
    transformed_result, result_modified = transform_preorder(
        inner_comp.result, transform
    )
    if not (modified or values_modified or result_modified):
      return inner_comp, False
    return (
        building_blocks.Block(transformed_variables, transformed_result),
        True,
    )
  else:
    raise NotImplementedError(
        f'Unexpected building block found: {type(inner_comp)}.'
    )


def transform_postorder_with_symbol_bindings(
    comp: building_blocks.ComputationBuildingBlock,
    transform: Callable[
        [building_blocks.ComputationBuildingBlock, 'SymbolTree'],
        building_blocks.ComputationBuildingBlock,
    ],
    symbol_tree: 'SymbolTree',
):
  """Uses symbol binding hooks to execute transformations.

  `transform_postorder_with_symbol_bindings` hooks into the preorder traversal
  that is defined by walking down the tree to its leaves, using
  the variable bindings along this path to push information onto
  the given `SymbolTree`. Once we hit the leaves, we walk back up the
  tree in a postorder fashion, calling `transform` as we go.

  The transformations `transform_postorder_with_symbol_bindings` executes are
  therefore stateful in some sense. Here 'stateful' means that a transformation
  executed on a given AST node in general depends on not only the node itself
  or its immediate vicinity; possibly there is some global information on which
  this transformation depends. `transform_postorder_with_symbol_bindings` is
  functional 'from AST to AST' (where `comp` represents the root of an AST) but
  not 'from node to node'.

  One important fact to note: there are recursion invariants that
  `transform_postorder_with_symbol_bindings` uses the `SymbolTree` data
  structure to enforce. In particular, within a `transform` call the following
  invariants hold:

  * `symbol_tree.update_payload_with_name` with an argument `name` will call
    `update` on the `BoundVariableTracker` in `symbol_tree` which tracks the
    value of `ref` active in the current lexical scope. Will raise a
    `NameError` if none exists.

  * `symbol_tree.get_payload_with_name` with a string argument `name` will
    return the `BoundVariableTracker` instance from `symbol_tree` which
    corresponds to the computation bound to the variable `name` in the current
    lexical scope. Will raise a `NameError` if none exists.

  These recursion invariants are enforced by the framework, and should be
  relied on when designing new transformations that depend on variable
  bindings.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to read
      information from or transform.
    transform: Python function accepting `comp` and `symbol_tree` arguments and
      returning `transformed_comp`.
    symbol_tree: Instance of `SymbolTree`, the data structure into which we may
      read information about variable bindings, and from which we may read.

  Returns:
    Returns a possibly modified version of `comp`, an instance
    of `building_blocks.ComputationBuildingBlock`, along with a
    Boolean with the value `True` if `comp` was transformed and `False` if it
    was not.
  """
  identifier_seq = itertools.count(start=1)

  def _transform_postorder_with_symbol_bindings_switch(
      comp, transform, ctxt_tree, identifier_sequence
  ):
    """Recursive helper function delegated to after binding comp_id sequence."""
    if isinstance(
        comp,
        (
            building_blocks.CompiledComputation,
            building_blocks.Data,
            building_blocks.Intrinsic,
            building_blocks.Literal,
            building_blocks.Placement,
            building_blocks.Reference,
        ),
    ):
      return _traverse_leaf(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, building_blocks.Selection):
      return _traverse_selection(
          comp, transform, ctxt_tree, identifier_sequence
      )
    elif isinstance(comp, building_blocks.Struct):
      return _traverse_tuple(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, building_blocks.Call):
      return _traverse_call(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, building_blocks.Lambda):
      return _traverse_lambda(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, building_blocks.Block):
      return _traverse_block(comp, transform, ctxt_tree, identifier_sequence)
    else:
      raise NotImplementedError(
          f'Unexpected building block found: {type(comp)}.'
      )

  def _traverse_leaf(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for leaf nodes."""
    _ = next(identifier_seq)
    return transform(comp, context_tree)

  def _traverse_selection(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for selection nodes."""
    _ = next(identifier_seq)
    source, source_modified = _transform_postorder_with_symbol_bindings_switch(
        comp.source, transform, context_tree, identifier_seq
    )
    if source_modified:
      # Normalize selection to index based on the type signature of the
      # original source. The new source may not have names present.
      if comp.index is not None:
        index = comp.index
      else:
        names = [n for n, _ in comp.source.type_signature.items()]
        index = names.index(comp.name)
      comp = building_blocks.Selection(source, index=index)
    comp, comp_modified = transform(comp, context_tree)
    return comp, comp_modified or source_modified

  def _traverse_tuple(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for tuple nodes."""
    _ = next(identifier_seq)
    elements = []
    elements_modified = False
    for key, value in comp.items():
      value, value_modified = _transform_postorder_with_symbol_bindings_switch(
          value, transform, context_tree, identifier_seq
      )
      elements.append((key, value))
      elements_modified = elements_modified or value_modified
    if elements_modified:
      comp = building_blocks.Struct(elements)
    comp, comp_modified = transform(comp, context_tree)
    return comp, comp_modified or elements_modified

  def _traverse_call(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for call nodes."""
    _ = next(identifier_seq)
    fn, fn_modified = _transform_postorder_with_symbol_bindings_switch(
        comp.function, transform, context_tree, identifier_seq
    )
    if comp.argument is not None:
      arg, arg_modified = _transform_postorder_with_symbol_bindings_switch(
          comp.argument, transform, context_tree, identifier_seq
      )
    else:
      arg, arg_modified = (None, False)
    if fn_modified or arg_modified:
      comp = building_blocks.Call(fn, arg)
    comp, comp_modified = transform(comp, context_tree)
    return comp, comp_modified or fn_modified or arg_modified

  def _traverse_lambda(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for lambda nodes."""
    comp_id = next(identifier_seq)
    context_tree.drop_scope_down(comp_id)
    context_tree.ingest_variable_binding(name=comp.parameter_name, value=None)
    result, result_modified = _transform_postorder_with_symbol_bindings_switch(
        comp.result, transform, context_tree, identifier_seq
    )
    context_tree.walk_to_scope_beginning()
    if result_modified:
      comp = building_blocks.Lambda(
          comp.parameter_name, comp.parameter_type, result
      )
    comp, comp_modified = transform(comp, context_tree)
    context_tree.pop_scope_up()
    return comp, comp_modified or result_modified

  def _traverse_block(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for block nodes."""
    comp_id = next(identifier_seq)
    context_tree.drop_scope_down(comp_id)
    variables = []
    variables_modified = False
    for key, value in comp.locals:
      value, value_modified = _transform_postorder_with_symbol_bindings_switch(
          value, transform, context_tree, identifier_seq
      )
      context_tree.ingest_variable_binding(name=key, value=value)
      variables.append((key, value))
      variables_modified = variables_modified or value_modified
    result, result_modified = _transform_postorder_with_symbol_bindings_switch(
        comp.result, transform, context_tree, identifier_seq
    )
    context_tree.walk_to_scope_beginning()
    if variables_modified or result_modified:
      comp = building_blocks.Block(variables, result)
    comp, comp_modified = transform(comp, context_tree)
    context_tree.pop_scope_up()
    return comp, comp_modified or variables_modified or result_modified

  return _transform_postorder_with_symbol_bindings_switch(
      comp, transform, symbol_tree, identifier_seq
  )


class BoundVariableTracker(abc.ABC):
  """Abstract class representing a mutable variable binding."""

  def __init__(
      self, name: str, value: Optional[building_blocks.ComputationBuildingBlock]
  ):
    """Initializes `BoundVariableTracker`.

    The initializer is likely to be overwritten by subclasses in order to
    attach more state to the `BoundVariableTracker`. Each of them must
    satisfy the same interface, however. This is simply because the
    `BoundVariableTracker` represents a variable binding in an AST;
    no more information is avaiable to it than the `name`-`value` pair
    being bound together.

    Args:
      name: String name of variable to be bound.
      value: Value to bind to this name. Can be instance of
        `building_blocks.ComputationBuildingBlock` if this
        `BoundVariableTracker` represents a concrete binding to a variable (e.g.
        in a block locals declaration), or `None`, if this
        `BoundVariableTracker` represents merely a variable declaration (e.g. in
        a lambda).
    """
    self.name = name
    self.value = value

  def update(self, value=None):
    """Defines the way information is read into this node.

    Defaults to no-op.

    Args:
      value: Similar to `value` argument in initializer.
    """
    del value  # Unused

  @abc.abstractmethod
  def __str__(self):
    """Abstract string method required as context tree will delegate."""

  def __eq__(self, other):
    """Base class equality checks names and values equal."""
    if self is other:
      return True
    if not isinstance(other, BoundVariableTracker):
      return NotImplemented
    if self.name != other.name:
      return False
    if isinstance(
        self.value, building_blocks.ComputationBuildingBlock
    ) and isinstance(other.value, building_blocks.ComputationBuildingBlock):
      return (
          self.value.compact_representation()
          == other.value.compact_representation()
          and self.value.type_signature.is_equivalent_to(
              other.value.type_signature
          )
      )
    return self.value is other.value

  def __ne__(self, other):
    """Implementing __ne__ to enforce in Python2 the Python3 standard."""
    return not self == other


class _BeginScopePointer(BoundVariableTracker):
  """Sentinel representing the beginning of a scope defined by an AST node."""

  def __init__(self, name=None, value=None):
    if name is not None or value is not None:
      raise ValueError(
          "Please don't pass a name or value to "
          '_BeginScopePointer; it will simply be ignored.'
      )
    super().__init__('BeginScope', None)

  def update(self, value=None):
    del value  # Unused.
    raise RuntimeError("We shouldn't be trying to update the outer context.")

  def __str__(self):
    return self.name

  def __eq__(self, other):
    """Returns `True` iff `other` is also a `_BeginScopePointer`.

    Args:
      other: Value for equality comparison.

    Returns:
      Returns true iff `other` is also an instance of `_BeginScopePointer`.
    """
    # Using explicit type comparisons here to prevent a subclass from passing.
    # pylint: disable=unidiomatic-typecheck
    return type(other) is _BeginScopePointer
    # pylint: enable=unidiomatic-typecheck


class SymbolTree:
  """Data structure to hold variable bindings as we walk an AST.

  `SymbolTree` is designed to be constructed and mutatated as we traverse an
  AST, maintaining a pointer to an active node representing the variable
  bindings we currently have available as we walk the AST.

  `SymbolTree` is a hierarchical tree-like data structure. Its levels
  correspond to nodes in the AST it is tracking, meaning that walking into
  or out of a scope-defining node (a block or lambda) corresponds to
  moving up or down a level in the `SymbolTree`. Block constructs (a.k.a.
  the let statement) binds variables sequentially, and this sequential binding
  corresponds to variables bound at the same level of the `SymbolTree`.

  Each instance of the node class can be used at most once in the symbol tree,
  as checked by memory location. This disallows circular tree structures that
  could cause an infinite loop in recursive equality testing or printing.
  """

  def __init__(self, payload_type: type[BoundVariableTracker]):
    """Initializes `SymbolTree` with its payload type.

    Args:
      payload_type: Class which subclasses BoundVariableTracker; the type of
        payloads to be constructed and held in this SymbolTree.
    """
    initial_node = SequentialBindingNode(_BeginScopePointer())
    self.active_node = initial_node
    self.payload_type = payload_type
    self._node_ids = {id(initial_node): 1}

  def get_payload_with_name(self, name: str) -> Optional[BoundVariableTracker]:
    """Returns payload corresponding to `name` in active variable bindings.

    Note that this method obeys `dict.get`-like semantics; instead of raising
    when asked to address an unbound name, it simply returns `None`.

    Args:
      name: String name to find in currently active context.

    Returns:
      Returns instance of `BoundVariableTracker` corresponding to `name`
      in context represented by `active_comp`, or `None` if the requested
      name is unbound in the current context.
    """
    comp = typing.cast(SequentialBindingNode, self.active_node)
    while comp.parent is not None or comp.older_sibling is not None:
      if name == comp.payload.name:
        return comp.payload
      if comp.older_sibling is not None:
        comp = comp.older_sibling
      elif comp.parent is not None:
        comp = comp.parent
    return None

  def get_higher_payloads_with_value(self, value, equal_fn=None):
    """Returns payloads above `active_node` whose `value` is equal to `value`.

    Args:
      value: The value to test.
      equal_fn: The optional function to use to determine equality, if `None` is
        specified `operator.is_` is used.
    """
    payloads = []
    if equal_fn is None:
      equal_fn = operator.is_
    node = typing.cast(SequentialBindingNode, self.active_node)
    while node.parent is not None or node.older_sibling is not None:
      if node.payload.value is not None and equal_fn(value, node.payload.value):
        payloads.append(node.payload)
      if node.older_sibling is not None:
        node = node.older_sibling
      elif node.parent is not None:
        node = node.parent
    return payloads

  def update_payload_with_name(self, name: str):
    """Calls `update` if `name` is found among the available symbols.

    If there is no such available symbol, simply does nothing.

    Args:
      name: A string; generally, this is the variable a walker has encountered
        in a AST, and which it is relying on `SymbolTable` to address correctly.

    Raises:
      ValueError: If `name` is not found among the bound names currently
        available in `self`.
    """
    comp = typing.cast(SequentialBindingNode, self.active_node)
    while comp.parent is not None or comp.older_sibling is not None:
      if name == comp.payload.name:
        comp.payload.update(name)
        return None
      if comp.older_sibling is not None:
        comp = comp.older_sibling
      elif comp.parent is not None:
        comp = comp.parent
    raise ValueError(
        "The name '{}' is not available in '{}'.".format(name, self)
    )

  def walk_to_scope_beginning(self):
    """Walks `active_node` back to the sentinel node beginning current scope.

    `walk_to_scope_beginning` resolves the issue of scope at a node which
    introduces scope in the following manner: each of these nodes (for instance,
    a `building_blocks.Lambda`) corresponds to a sentinel value of
    the `_BeginScopePointer` class, ensuring that these nodes do not have access
    to
    scope that is technically not available to them. That is, we conceptualize
    the node corresponding to `(x -> x)` as existing in the scope outside of the
    binding of `x`, and therefore is unable to reference `x`. However, these
    nodes can walk down their variable declarations via
    `walk_down_one_variable_binding` in order to inspect these declarations and
    perhaps execute some logic based on them.
    """
    scope_sentinel = _BeginScopePointer()
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    assert self.active_node is not None
    while self.active_node.payload != scope_sentinel:
      self.active_node = self.active_node.older_sibling
      assert self.active_node is not None

  def pop_scope_up(self):
    """Moves `active_node` up one level in the `SymbolTree`.

    Raises:
      Raises ValueError if we are already at the highest level.
    """
    self.walk_to_scope_beginning()
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.parent:
      self.active_node = self.active_node.parent
    else:
      raise ValueError(
          'You have tried to pop out of the highest level in this `SymbolTree`.'
      )

  def drop_scope_down(self, comp_id: int) -> None:
    """Constructs a new scope level for `self`.

    Scope levels in `SymbolTree` correspond to scope-introducing nodes in ASTs;
    that is, either `building_blocks.Block` or `building_blocks.Lambda` nodes.
    Inside of these levels, variables are bound in sequence. The implementer of
    a transformation function needing to interact with scope should never need
    to explicitly walk the scope levels `drop_scope_down` constructs;
    `drop_scope_down` is simply provided for ease of exposing to a traversal
    function.

    Args:
      comp_id: Integer representing a unique key for the
        `building_blocks.ComputationBuildingBlock` which is defines this scope.
        Used to differentiate between scopes which both branch from the same
        point in the tree.
    """
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.children.get(comp_id) is None:
      node = SequentialBindingNode(_BeginScopePointer())
      self._add_child(comp_id, node)
      self._move_to_child(comp_id)
    else:
      self._move_to_child(comp_id)

  def walk_down_one_variable_binding(self):
    """Moves `active_node` to the younger sibling of the current active node.

    This action represents walking from one variable binding in the
    `SymbolTree` to the next, sequentially.

    If there is no such variable binding, then the lower bound variables must
    be accessed via `drop_scope_down`.

    Raises:
      Raises ValueError if there is no such available variable binding.
    """
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.younger_sibling:
      self.active_node = self.active_node.younger_sibling
    else:
      raise ValueError(
          'You have tried to move to a nonexistent variable binding in {}'
          .format(self)
      )

  def ingest_variable_binding(
      self,
      name: Optional[str],
      value: Optional[building_blocks.ComputationBuildingBlock],
  ) -> None:
    """Constructs or updates node in symbol tree as AST is walked.

    Passes `name` and `value` onto the symbol tree's node constructor, with
    `mode` determining how the node being constructed or updated
    relates to the symbol tree's `active_node`.

    If there is no preexisting node in the symbol tree bearing the
    requested relationship to the active node, a new one will be constructed and
    initialized. If there is an existing node, `ingest_variable_binding` checks
    that this node has the correct `payload.name`, and overwrites its
    `payload.value` with the `value` argument.

    Args:
      name: The string name of the `CompTracker` instance we are constructing or
        updating.
      value: Instance of `building_blocks.ComputationBuildingBlock` or `None`,
        as in the `value` to pass to symbol tree's node payload constructor.

    Raises:
      ValueError: If we are passed a name-mode pair such that a
        preexisting node in the symbol tree bears this relationship with
        the active node, but has a different name. This is an indication
        that either a transformation has failed to happen in the symbol tree
        or that we have a symbol tree instance that does not match the
        computation we are currently processing.
    """
    if (name is None or not name) and value is None:
      return

    node = SequentialBindingNode(self.payload_type(name=name, value=value))
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.younger_sibling is None:
      self._add_younger_sibling(node)
      self.walk_down_one_variable_binding()
    else:
      if self.active_node.younger_sibling.payload.name != name:
        raise ValueError(
            'You have a mismatch between your symbol tree and the '
            'computation you are trying to process; your symbol tree is {} '
            'and you are looking for a BoundVariableTracker with name {} '
            'and value {}'.format(self, name, value)
        )
      self.walk_down_one_variable_binding()
      self.active_node.payload.value = value

  def _add_younger_sibling(self, comp_tracker: 'SequentialBindingNode') -> None:
    """Appends comp as younger sibling of current `active_node`."""
    if self._node_ids.get(id(comp_tracker)):
      raise ValueError(
          'Each instance of {} can only appear once in a given symbol tree.'
          .format(self.payload_type)
      )
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.younger_sibling is not None:
      raise ValueError('Ambiguity in adding a younger sibling')
    comp_tracker.set_older_sibling(self.active_node)
    self.active_node.set_younger_sibling(comp_tracker)
    self._node_ids[id(comp_tracker)] = 1

  def _add_child(
      self,
      constructing_comp_id: int,
      comp_tracker: 'SequentialBindingNode',
  ) -> None:
    """Writes `comp_tracker` to children of active node.

    Each `SequentialBindingNode` keeps a `dict` of its children; `_add_child`
    updates the value of this `dict` with key `constructing_comp_id` to be
    `comp_tracker`.

    Notice that `constructing_comp_id` is simply a way of addressing the
    children in this dict; it is not necessarily globally unique, as long
    as it is sufficient to address child scopes.

    Args:
      constructing_comp_id: Key to identify child being constructed from the
        parent scope.
      comp_tracker: Instance of `SequentialBindingNode`, the node to add as a
        child of `active_node`.
    """
    if self._node_ids.get(id(comp_tracker)):
      raise ValueError(
          'Each node can only appear once in a given'
          'symbol tree. You have tried to add {} '
          'twice.'.format(comp_tracker.payload)
      )
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    comp_tracker.set_parent(self.active_node)
    self.active_node.add_child(constructing_comp_id, comp_tracker)
    self._node_ids[id(comp_tracker)] = 1

  def _move_to_child(self, comp_id):
    """Moves `active_node` to child of current active node with key `comp_id`.

    Args:
      comp_id: Integer representing the position of the child we wish to update
        `active_node` to point to in a preorder traversal of the AST.

    Raises:
      ValueError: If the active node has no child with the correct id.
    """
    self.active_node = typing.cast(SequentialBindingNode, self.active_node)
    if self.active_node.children.get(comp_id) is not None:
      self.active_node = self.active_node.get_child(comp_id)
    else:
      raise ValueError('You have tried to move to a nonexistent child.')

  def _equal_under_node(self, self_node, other_node):
    """Recursive helper function to check equality of `SymbolTree`s."""
    if self_node is None and other_node is None:
      return True
    if self_node is None or other_node is None:
      return False
    if self_node.payload != other_node.payload:
      return False
    if len(self_node.children) != len(other_node.children):
      return False
    for (_, val_1), (_, val_2) in zip(
        self_node.children.items(), other_node.children.items()
    ):
      # keys not compared to avoid coupling walking logic to `SymbolTree`.
      if not self._equal_under_node(val_1, val_2):
        return False
    return self._equal_under_node(
        self_node.younger_sibling, other_node.younger_sibling
    )

  def __eq__(self, other):
    """Walks to root of `self` and `other` before testing equality of subtrees.

    Args:
      other: Instance of `SymbolTree` to test for equality with `self`.

    Returns:
      Returns `True` if and only if `self` and `other` are the same
      structurally (each node has the same number of children and siblings) and
      each node of `self` compares as equal with the node in the corresponding
      position of `other`.
    """
    if self is other:
      return True
    if not isinstance(other, SymbolTree):
      return NotImplemented
    self_root = _walk_to_root(self.active_node)
    other_root = _walk_to_root(other.active_node)
    return self._equal_under_node(self_root, other_root)

  def __ne__(self, other):
    return not self == other

  def _string_under_node(self, node: 'SequentialBindingNode') -> str:
    """Rescursive helper function to generate string reps of `SymbolTree`s."""
    if node is self.active_node:
      active_node_indicator = '*'
    else:
      active_node_indicator = ''
    symbol_tree_string = '[' + str(node.payload) + active_node_indicator + ']'
    if node.children:
      symbol_tree_string += '->{'
      for _, child_node in node.children.items():
        if not child_node.older_sibling:
          symbol_tree_string += '('
          symbol_tree_string += self._string_under_node(child_node)
          symbol_tree_string += '),('
      symbol_tree_string = symbol_tree_string[:-2]
      symbol_tree_string += '}'
    if node.younger_sibling:
      symbol_tree_string += '-' + self._string_under_node(node.younger_sibling)
    return symbol_tree_string

  def __str__(self):
    """Generates a string representation of this `SymbolTree`.

    First we walk up to the root node, then we walk down
    the tree generating string rep of this symbol tree.

    Returns:
      Returns a string representation of the current `SymbolTree`, with
      the node labeled the active node identified with a *.
    """
    node = self.active_node
    root_node = _walk_to_root(node)
    return self._string_under_node(root_node)


def _walk_to_root(node):
  while node.parent is not None or node.older_sibling is not None:
    while node.older_sibling is not None:
      node = node.older_sibling
    while node.parent is not None:
      node = node.parent
  return node


class SequentialBindingNode:
  """Represents a node in a context tree with sequential-binding semantics.

  `SequentialBindingNode`s are designed to be constructed and pushed into
  a context tree as an AST representing a given computation is walked.

  Each `SequentialBindingNode` holds as payload a variable binding in the AST.
  The node-node relationships encoded by the `SequentialBindingNode` data
  structure determine how the context tree must be walked in order to resolve
  variables and track their values in the AST.

  Parent-child relationships represent relationships between levels of the AST,
  meaning, moving through an AST node which defines a variable scope in preorder
  corresponds to moving from a `SequentialBindingNode` to one of its children,
  and moving through such a node postorder corresponds to moving from a
  `SequentialBindingNode` to its parent.

  Sibling-sibling relationships are particular to sequential binding of
  variables in `building_blocks.Block` constructs; binding
  a new variable in such a construct corresponds to moving from a
  `SequentialBindingNode` to its (unique) younger sibling.
  """

  def __init__(self, payload: BoundVariableTracker):
    """Initializes `SequentialBindingNode`.

    Args:
      payload: Instance of BoundVariableTracker representing the payload of this
        node.
    """
    self.payload = payload
    self._children = collections.OrderedDict()
    self._parent = None
    self._older_sibling = None
    self._younger_sibling = None

  @property
  def parent(self):
    return self._parent

  @property
  def children(self):
    return self._children

  @property
  def older_sibling(self):
    return self._older_sibling

  @property
  def younger_sibling(self):
    return self._younger_sibling

  def set_parent(self, node: 'SequentialBindingNode') -> None:
    """Sets the _parent scope of `self` to the binding embodied by `node`.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` to set as parent of `self`.
    """
    self._parent = node

  def set_older_sibling(self, node: 'SequentialBindingNode') -> None:
    """Sets the older sibling scope of `self` to `node`.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` to set as older sibling of
        `self`.
    """
    self._older_sibling = node

  def set_younger_sibling(self, node: 'SequentialBindingNode') -> None:
    """Sets the younger sibling scope of `self` to `node`.

    This corresponds to binding a new variable in a
    `building_blocks.Block` construct.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` representing this new binding.
    """
    self._younger_sibling = node

  def add_child(self, comp_id: int, node: 'SequentialBindingNode') -> None:
    """Sets the child scope of `self` indexed by `comp_id` to `node`.

    This corresponds to encountering a node in an AST which defines a variable
    scope.

    If a child with this `comp_id` already exists, it is replaced, as in a
    `dict`.

    Args:
      comp_id: The identifier of the computation generating this scope.
      node: Instance of `SequentialBindingNode` representing this new binding.
    """
    self._children[comp_id] = node

  def get_child(self, comp_id):
    """Returns the child of `self` identified by `comp_id` if one exists.

    Args:
      comp_id: Integer used to address child of `self` by position of
        corresponding AST node in a preorder traversal of the AST.

    Returns:
      Instance of `SequentialBindingNode` if an appropriate child of `self`
      exists, or `None`.
    """
    return self._children.get(comp_id)


class ReferenceCounter(BoundVariableTracker):
  """Data container to track number References to a variable in an AST.

  Attributes:
    name: The string name representing the variable whose binding is represented
      by an instance of `ReferenceCounter`.
    value: The value bound to `name`. Can be an instance of
      `building_blocks.ComputationBuildingBlock` or None if this binding is
      simply a placeholder, e.g. in a Lambda.
    count: An integer tracking how many times the variable an instance of
      `ReferenceCounter` represents is referenced in an AST.
  """

  def __init__(self, name, value):
    super().__init__(name, value)
    self.count = 0

  def update(self, value=None):
    del value  # Unused.
    self.count += 1

  def __str__(self):
    return 'Instance count: {}; value: {}; name: {}.'.format(
        self.count, self.value, self.name
    )

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, ReferenceCounter):
      return NotImplemented
    if not super().__eq__(other):
      return False
    return self.count == other.count


def get_unique_names(
    comp: building_blocks.ComputationBuildingBlock,
) -> set[str]:
  """Returns the unique names bound or referred to in `comp`."""
  names = set()

  def _update(comp):
    if isinstance(comp, building_blocks.Block):
      names.update([name for name, _ in comp.locals])
    elif isinstance(comp, building_blocks.Lambda):
      if comp.parameter_type is not None:
        names.add(comp.parameter_name)
    elif isinstance(comp, building_blocks.Reference):
      names.add(comp.name)
    return comp, False

  transform_postorder(comp, _update)
  return names


def get_map_of_unbound_references(
    comp: building_blocks.ComputationBuildingBlock,
) -> dict[building_blocks.ComputationBuildingBlock, set[str]]:
  """Gets a Python `dict` of unbound references in `comp`, keyed by Python `id`.

  Computations that are equal will have the same collections of unbounded
  references, so it is safe to use `comp` as the key for this `dict` even though
  a given computation may appear in many positions in the AST.

  Args:
    comp: The computation building block to parse.

  Returns:
    A Python `dict` of elements where keys are the computations in `comp` and
    values are a Python `set` of the names of the unbound references in the
    subtree of that computation.
  """
  references = {}

  def _update(comp):
    """Updates the Python dict of references."""
    if isinstance(comp, building_blocks.Reference):
      references[comp] = set((comp.name,))
    elif isinstance(comp, building_blocks.Block):
      references[comp] = set()
      names = []
      for name, variable in comp.locals:
        elements = references[variable]
        references[comp].update([e for e in elements if e not in names])
        names.append(name)
      elements = references[comp.result]
      references[comp].update([e for e in elements if e not in names])
    elif isinstance(comp, building_blocks.Call):
      elements = references[comp.function].copy()
      if comp.argument is not None:
        elements.update(references[comp.argument])
      references[comp] = elements
    elif isinstance(comp, building_blocks.Lambda):
      elements = references[comp.result]
      references[comp] = set([e for e in elements if e != comp.parameter_name])
    elif isinstance(comp, building_blocks.Selection):
      references[comp] = references[comp.source]
    elif isinstance(comp, building_blocks.Struct):
      elements = [references[e] for e in comp]
      references[comp] = set(itertools.chain.from_iterable(elements))
    else:
      references[comp] = set()
    return comp, False

  transform_postorder(comp, _update)
  return references
