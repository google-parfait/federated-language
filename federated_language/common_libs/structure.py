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
"""Container for structures with named and/or unnamed fields."""

import collections
from collections.abc import Callable, Iterable, Iterator, Mapping
import typing
from typing import Generic, Optional, TypeVar, Union

import attrs
from federated_language.common_libs import py_typecheck
import tree


_T = TypeVar('_T')
_U = TypeVar('_U')


class Struct(Generic[_T]):
  """Represents a struct-like structure with named and/or unnamed fields.

  `Struct`s are similar to `collections.namedtuple` in that their elements can
  be accessed by name or by index. However, `Struct`s provide a performance
  improvement over `collections.namedtuple` by using a single class to
  represent values with many different possible structures, rather than
  creating a brand new class for every new instance.

  `Struct`s are commonly used inside Tensorflow Federated as a standard
  intermediate representation of other structure types, including `list`s,
  `tuple`s, `dict`s, `namedtuple`s, and `attr.s` classes.

  Example:

  ```python
  x = Struct([('foo', 10), (None, 20), ('bar', 30)])

  len(x) == 3
  x[0] == 10
  x[1] == 20
  x[2] == 30
  list(iter(x)) == [10, 20, 30]
  dir(x) == ['bar', 'foo']
  x.foo == 10
  x['bar'] == 30
  ```

  Note that field names are optional, allowing `Struct` to be used like an
  ordinary positional tuple.
  """

  __slots__ = (
      '_hash',
      '_element_array',
      '_name_to_index',
      '_name_array',
      '_elements_cache',
  )

  @classmethod
  def named(cls, **kwargs: _T) -> 'Struct':
    """Constructs a new `Struct` with all named elements."""
    return cls(tuple(kwargs.items()))

  @classmethod
  def unnamed(cls, *args: _T) -> 'Struct':
    """Constructs a new `Struct` with all unnamed elements."""
    return cls(tuple((None, v) for v in args))

  def __init__(self, elements: Iterable[tuple[Optional[str], _T]]):
    """Constructs a new `Struct` with the given elements.

    Args:
      elements: An iterable of element specifications, each being a pair
        consisting of the element name (either `str`, or `None`), and the
        element value. The order is significant.

    Raises:
      TypeError: if the `elements` are not a list, or if any of the items on
        the list is not a pair with a string at the first position.
    """
    values = []
    names = []
    name_to_index = {}
    reserved_names = frozenset(('_asdict',) + Struct.__slots__)
    for idx, e in enumerate(elements):
      if not py_typecheck.is_name_value_pair(e):
        raise TypeError(
            'Expected every item on the list to be a pair in which the first '
            'element is a string, found {!r}.'.format(e)
        )
      name, value = e
      if name in reserved_names:
        raise ValueError(
            'The names in {} are reserved. You passed the name {}.'.format(
                reserved_names, name
            )
        )
      elif name in name_to_index:
        raise ValueError(
            '`Struct` does not support duplicated names, found {}.'.format(
                [e[0] for e in elements]
            )
        )
      names.append(name)
      values.append(value)
      if name is not None:
        name_to_index[name] = idx
    self._element_array = tuple(values)
    self._name_to_index = name_to_index
    self._name_array = names
    self._hash = None
    self._elements_cache = None

  def _elements(self) -> list[tuple[Optional[str], _T]]:
    if self._elements_cache is None:
      self._elements_cache = list(zip(self._name_array, self._element_array))
    return self._elements_cache

  def __len__(self) -> int:
    return len(self._element_array)

  def __iter__(self) -> Iterator[_T]:
    return iter(self._element_array)

  def __dir__(self) -> list[str]:
    """The list of names.

    IMPORTANT: `len(self)` may be greater than `len(dir(self))`, since field
    names are not required by `Struct`.

    IMPORTANT: the Python `dir()` built-in sorts the list returned by this
    method.

    Returns:
      A `list` of `str`.
    """
    return list(self._name_to_index.keys())

  def __getitem__(self, key: Union[int, str, slice]) -> _T:
    if isinstance(key, str):
      return self.__getattr__(key)
    if isinstance(key, int):
      if key < 0 or key >= len(self._element_array):
        raise IndexError(
            'Element index {} is out of range, `Struct` has {} elements.'
            .format(key, len(self._element_array))
        )
    return self._element_array[key]

  def __getattr__(self, name: str) -> _T:
    if name not in self._name_to_index:
      raise AttributeError(
          'The `Struct` of length {:d} does not have named field "{!s}". '
          'Fields (up to first 10): {!s}'.format(
              len(self._element_array),
              name,
              list(self._name_to_index.keys())[:10],
          )
      )
    return self._element_array[self._name_to_index[name]]

  def __setstate__(self, state):
    # IMPORTANT: At unpickling time, `__getattr__()` is called by the
    # implementation of `object.__reduce__()`. However, `__init__()` is not
    # called when unpickling an instance. This class implements `__getattr__()`
    # and accesses attributes that have not been set. Implementing
    # `__setstate__` instead of relying on `object.__reduce__()` prevents this.
    dict_attributes, slot_attributes = state
    self.__dict__.update(dict_attributes)
    for name, value in slot_attributes.items():
      setattr(self, name, value)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, Struct):
      return NotImplemented
    return (
        self._element_array,
        self._name_array,
    ) == (
        other._element_array,
        other._name_array,
    )

  def __ne__(self, other: object) -> bool:
    return not self == other

  def __repr__(self) -> str:
    return 'Struct([{}])'.format(
        ', '.join('({!r}, {!r})'.format(n, v) for n, v in iter_elements(self))
    )

  def __str__(self) -> str:
    def _element_str(element: tuple[Optional[str], object]) -> str:
      name, value = element
      if name is not None:
        return '{}={}'.format(name, value)
      return str(value)

    return '<{}>'.format(','.join(_element_str(e) for e in iter_elements(self)))

  def __hash__(self) -> int:
    if self._hash is None:
      self._hash = hash((
          'Struct',  # salting to avoid type mismatch.
          self._element_array,
          tuple(self._name_array),
      ))
    return self._hash

  def _asdict(
      self, recursive: bool = False
  ) -> collections.OrderedDict[str, _T]:
    """Returns an `collections.OrderedDict` mapping field names to their values.

    Args:
      recursive: Whether to convert nested `Struct`s recursively.
    """
    return to_odict(self, recursive=recursive)


def name_list(struct: Struct) -> list[str]:
  """Returns a `list` of the names of the named fields in `struct`.

  Args:
    struct: An instance of `Struct`.

  Returns:
    The list of string names for the fields that are named. Names appear in
    order, skipping names that are `None`.
  """
  names = struct._name_array  # pylint: disable=protected-access
  return [n for n in names if n is not None]


def name_list_with_nones(struct: Struct) -> list[Optional[str]]:
  """Returns an iterator over the names of all fields in `struct`."""
  return struct._name_array  # pylint: disable=protected-access


def to_elements(struct: Struct[_T]) -> list[tuple[Optional[str], _T]]:
  """Retrieves the list of (name, value) pairs from a `Struct`.

  Modeled as a module function rather than a method of `Struct` to avoid
  naming conflicts with the tuple attributes, and so as not to expose the user
  to this implementation-oriented functionality.

  Args:
    struct: An instance of `Struct`.

  Returns:
    The list of (name, value) pairs in which names can be None. Identical to
    the format that's accepted by the tuple constructor.

  Raises:
    TypeError: if the argument is not an `Struct`.
  """
  return struct._elements().copy()  # pylint: disable=protected-access


def iter_elements(struct: Struct[_T]) -> Iterator[tuple[Optional[str], _T]]:
  """Returns an iterator over (name, value) pairs from a `Struct`.

  Modeled as a module function rather than a method of `Struct` to avoid
  naming conflicts with the tuple attributes, and so as not to expose the user
  to this implementation-oriented functionality.

  Args:
    struct: An instance of `Struct`.

  Returns:
    An iterator of 2-tuples of name, value pairs, representing the elements of
      `struct`.

  Raises:
    TypeError: if the argument is not an `Struct`.
  """
  return iter(struct._elements())  # pylint: disable=protected-access


def to_odict(
    struct: Struct[_T], recursive: bool = False
) -> collections.OrderedDict[str, _T]:
  """Returns `struct` as an `collections.OrderedDict`, if possible.

  Args:
    struct: An `Struct`.
    recursive: Whether to convert nested `Struct`s recursively.

  Raises:
    ValueError: If the `Struct` contains unnamed elements.
  """

  def _to_odict(
      elements: list[tuple[Optional[str], _T]],
  ) -> collections.OrderedDict[str, _T]:
    for name, _ in elements:
      if name is None:
        raise ValueError(
            'Cannot convert an `Struct` with unnamed entries to a '
            '`collections.OrderedDict`: {}'.format(struct)
        )
    elements = typing.cast(list[tuple[str, _T]], elements)
    return collections.OrderedDict(elements)

  if recursive:
    return _to_container_recursive(struct, _to_odict)
  else:
    return _to_odict(to_elements(struct))


def to_odict_or_tuple(
    struct: Struct[_T], recursive: bool = True
) -> Union[collections.OrderedDict[str, _T], tuple[_T, ...]]:
  """Returns `struct` as an `collections.OrderedDict` or `tuple`, if possible.

  If all elements of `struct` have names, convert `struct` to an
  `collections.OrderedDict`. If no element has a name, convert `struct` to a
  `tuple`. If
  `struct` has both named and unnamed elements, raise an error.

  Args:
    struct: A `Struct`.
    recursive: Whether to convert nested `Struct`s recursively.

  Raises:
    ValueError: If `struct` (or any nested `Struct` when `recursive=True`)
      contains both named and unnamed elements.
  """

  def _to_odict_or_tuple(
      elements: list[tuple[Optional[str], _T]],
  ) -> Union[collections.OrderedDict[str, _T], tuple[_T, ...]]:
    fields_are_named = tuple(name is not None for name, _ in elements)
    if any(fields_are_named):
      if not all(fields_are_named):
        raise ValueError(
            'Cannot convert a `Struct` with both named and unnamed '
            'entries to an collections.OrderedDict or tuple: {!r}'.format(
                struct
            )
        )
      elements = typing.cast(list[tuple[str, _T]], elements)
      return collections.OrderedDict(elements)
    else:
      return tuple(value for _, value in elements)

  if recursive:
    return _to_container_recursive(struct, _to_odict_or_tuple)
  else:
    return _to_odict_or_tuple(to_elements(struct))


def flatten(struct: object) -> list[object]:
  """Returns a list of values in a possibly recursively nested `Struct`.

  NOTE: This implementation is not compatible with the approach of
  `tf.nest.flatten`, which enforces lexical order for
  `collections.OrderedDict`s.

  Args:
    struct: A `Struct`, possibly recursively nested, or a non-`Struct` element
      that can be packed with `tf.nest.flatten`. If `struct` has
      non-`Struct`-typed fields which should be flattened further, they should
      not contain inner `Structs`, as these will not be flattened (e.g.
      `Struct([('a', collections.OrderedDict(b=Struct([('c', 5)])))])` would not
      be valid).

  Returns:
    The list of leaf values in the `Struct`.
  """
  if not isinstance(struct, Struct):
    return tree.flatten(struct)
  else:
    result = []
    for _, v in iter_elements(struct):
      result.extend(flatten(v))
    return result


def pack_sequence_as(
    structure: Struct[_T], flat_sequence: list[_T]
) -> Struct[_T]:
  """Returns a list of values in a possibly recursively nested `Struct`.

  Args:
    structure: A `Struct`, possibly recursively nested.
    flat_sequence: A flat Python list of values.

  Returns:
    A `Struct` nested the same way as `structure`, but with leaves
    replaced with `flat_sequence` such that when flatten, it yields a list
    with the same contents as `flat_sequence`.
  """

  def _pack(
      structure, flat_sequence: list[_T], position: int
  ) -> tuple[Struct[_T], int]:
    """Pack a leaf element or recurvisely iterate over an `Struct`."""
    if not isinstance(structure, Struct):
      # Ensure that our leaf values are not structures.
      if isinstance(
          structure, (list, dict, py_typecheck.SupportsNamedTuple)
      ) or attrs.has(type(structure)):
        raise TypeError(
            'Cannot pack sequence into type {!s}, only structures of '
            '`Struct` are supported, found a structure with types '
            '{!s}).'.format(type(structure), structure)
        )

      return flat_sequence[position], position + 1
    else:
      elements = []
      for k, v in iter_elements(structure):
        packed_v, position = _pack(v, flat_sequence, position)
        elements.append((k, packed_v))
      return Struct(elements), position

  result, _ = _pack(structure, flat_sequence, 0)
  # NOTE: trailing elements are currently ignored.
  return result


def _is_same_structure(a: Struct, b: Struct) -> bool:
  """Compares whether `a` and `b` have the same nested structure.

  This method is analogous to `tf.nest.assert_same_structure`,
  but returns a boolean rather than throwing an exception.

  Args:
    a: a `Struct` object.
    b: a `Struct` object.

  Returns:
    True iff `a` and `b` have the same nested structure.

  Raises:
    TypeError: if `a` or `b` are not of type `Struct`.
  """
  elems_a = to_elements(a)
  elems_b = to_elements(b)
  if len(elems_a) != len(elems_b):
    return False
  for elem_a, elem_b in zip(elems_a, elems_b):
    val_a = elem_a[1]
    val_b = elem_b[1]
    if elem_a[0] != elem_b[0]:
      return False
    if isinstance(val_a, Struct) and isinstance(val_b, Struct):
      return _is_same_structure(val_a, val_b)
    elif isinstance(val_a, Struct) or isinstance(val_b, Struct):
      return False
    else:
      try:
        tree.assert_same_structure(val_a, val_b, check_types=True)
      except (ValueError, TypeError):
        return False
  return True


def map_structure(fn: Callable[..., object], *structures: Struct) -> object:
  """Applies `fn` to each entry in `structure` and returns a new structure.

  This is a special implementation of `tf.nest.map_structure`
  that works for `Struct`.

  Args:
    fn: a callable that accepts as many arguments as there are structures.
    *structures: a scalar, tuple, or list of constructed scalars and/or
      tuples/lists, or scalars. NOTE: numpy arrays are considered scalars.

  Returns:
    A new structure with the same arity as `structure` and same type as
    `structure[0]`, whose values correspond to `fn(x[0], x[1], ...)` where
    `x[i]` is a value in the corresponding location in `structure[i]`.

  Raises:
    TypeError: if `fn` is not a callable, or *structure is not all `Struct` or
      all `tf.Tensor` typed values.
    ValueError: if `*structure` is empty.
  """
  if not structures:
    raise ValueError('Must provide at least one structure')

  if not all(isinstance(s, Struct) for s in structures):
    return tree.map_structure(fn, *structures)

  for i, other in enumerate(structures[1:]):
    if not _is_same_structure(structures[0], other):
      raise TypeError(
          'Structure at position {} is not the same structure'.format(i)
      )

  flat_structure = [flatten(s) for s in structures]
  entries = zip(*flat_structure)
  s = [fn(*x) for x in entries]

  return pack_sequence_as(structures[0], s)


def from_container(value: object, recursive=False) -> Struct:
  """Creates an instance of `Struct` from a Python container.

  By default, this conversion is only performed at the top level for Python
  dictionaries, `collections.OrderedDict`s, `namedtuple`s, `list`s,
  `tuple`s, and `attr.s` classes. Elements of these structures are not
  recursively converted.

  Args:
    value: _The Python container to convert.
    recursive: Whether to convert elements recursively (`False` by default).

  Returns:
    The corresponding instance of `Struct`.

  Raises:
    TypeError: If the `value` is not of one of the supported container types.
  """

  def _convert(
      value: object, recursive: bool, must_be_container: bool = False
  ) -> Struct:
    """The actual conversion function.

    Args:
      value: Same as in `from_container`.
      recursive: Same as in `from_container`.
      must_be_container: When set to `True`, causes an exception to be raised if
        `value` is not a container.

    Returns:
      The result of conversion.

    Raises:
      TypeError: If `value` is not a container and `must_be_container` has been
        set to `True`.
    """
    # TODO: b/224484886 - Downcasting to all handled types.
    value = typing.cast(
        Union[
            Struct,
            py_typecheck.SupportsNamedTuple,
            Mapping[str, object],
            dict[str, object],
            tuple[object, ...],
            list[object],
        ],
        value,
    )
    if isinstance(value, Struct):
      if recursive:
        return Struct((k, _convert(v, True)) for k, v in iter_elements(value))
      else:
        return value
    elif attrs.has(type(value)):
      return _convert(
          attrs.asdict(value, recurse=False), recursive, must_be_container
      )
    elif isinstance(value, py_typecheck.SupportsNamedTuple):
      return _convert(value._asdict(), recursive, must_be_container)
    elif isinstance(value, Mapping):
      items = value.items()
      if recursive:
        return Struct((k, _convert(v, True)) for k, v in items)
      else:
        return Struct(items)
    elif isinstance(value, (tuple, list)):
      if recursive:
        return Struct((None, _convert(v, True)) for v in value)
      else:
        return Struct((None, v) for v in value)
    elif must_be_container:
      raise TypeError(
          f'Unable to convert a Python object of type {type(value)} into an'
          f' `Struct`. Object: {value}'
      )
    else:
      return value

  return _convert(value, recursive, must_be_container=True)


def _to_container_recursive(
    value: Struct[_T],
    container_fn: Callable[[list[tuple[Optional[str], _T]]], _U],
) -> _U:
  """Recursively converts the `Struct` `value` to a new container type.

  This function is always recursive, since the non-recursive version would be
  just `container_fn(value)`.

  NOTE: _This function will only recurse through `Struct`s, so if called
  on the input `Struct([('a', 1), ('b', {'c': Struct(...)})])`
  the inner `Struct` will not be converted, because we do not recurse
  through Python `dict`s.

  Args:
    value: An `Struct`, possibly nested.
    container_fn: A function that takes a `list` of `(name, value)` tuples ( the
      elements of an `Struct`), and returns a new container holding the same
      values.

  Returns:
    A nested container of the type returned by `container_fn`.
  """

  def recurse(v):
    if isinstance(v, Struct):
      return _to_container_recursive(v, container_fn)
    else:
      return v

  return container_fn([(k, recurse(v)) for k, v in iter_elements(value)])


def has_field(structure: Struct, field: str) -> bool:
  """Returns `True` if the `structure` has the `field`.

  Args:
    structure: An instance of `Struct`.
    field: A string, the field to test for.
  """
  return field in structure._name_array  # pylint: disable=protected-access


def update_struct(structure, **kwargs):
  """Constructs a new `structure` with new values for fields in `kwargs`.

  This is a helper method for working structured objects in a functional manner.
  This method will create a new structure where the fields named by keys in
  `kwargs` replaced with the associated values.

  NOTE: _This method only works on the first level of `structure`, and does not
  recurse in the case of nested structures. A field that is itself a structure
  can be replaced with another structure.

  Args:
    structure: _The structure with named fields to update.
    **kwargs: _The list of key-value pairs of fields to update in `structure`.

  Returns:
    A new instance of the same type of `structure`, with the fields named
    in the keys of `**kwargs` replaced with the associated values.

  Raises:
    KeyError: If kwargs contains a field that is not in structure.
    TypeError: If structure is not a structure with named fields.
  """
  if not isinstance(
      structure, (Struct, Mapping, py_typecheck.SupportsNamedTuple)
  ) and not attrs.has(type(structure)):
    raise TypeError(
        '`structure` must be a structure with named fields (e.g. '
        'dict, attrs class, collections.namedtuple, '
        'federated_language.structure.Struct), but '
        'found {}'.format(type(structure))
    )
  if isinstance(structure, Struct):
    elements = [
        (k, v) if k not in kwargs else (k, kwargs.pop(k))
        for k, v in iter_elements(structure)
    ]
    if kwargs:
      raise KeyError(f'`structure` does not contain fields named {kwargs}')
    return Struct(elements)
  elif isinstance(structure, py_typecheck.SupportsNamedTuple):
    # In Python 3.8 and later `_asdict` no longer return OrdereDict, rather a
    # regular `dict`, so we wrap here to get consistent types across Python
    # version.s
    dictionary = structure._asdict()
  elif attrs.has(type(structure)):
    dictionary = attrs.asdict(structure, recurse=False)
  else:
    for key in kwargs:
      if key not in structure:
        raise KeyError(
            'structure does not contain a field named "{!s}"'.format(key)
        )
    # Create a copy to prevent mutation of the original `structure`
    dictionary = type(structure)(**structure)
  dictionary.update(kwargs)
  if isinstance(structure, Mapping):
    return dictionary
  return type(structure)(**dictionary)
