# Copyright 2024 Google LLC
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
"""Sphinx configuration.

See https://www.sphinx-doc.org/en/master/usage/configuration.html for more
information.
"""

import functools
import inspect
import os
import sys


# Project information
project = 'Federated Language'
copyright = '2024, Google LLC'  # pylint: disable=redefined-builtin
author = 'The Federated Language authors'
version = ''
release = ''

# General configuration
extensions = [
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
]
language = 'en'
master_doc = 'index'
pygments_style = 'sphinx'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ['_templates']

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'sticky_navigation': True,
}
html_context = {
    'display_github': True,
    'github_repo': 'https://github.com/google-parfait/federated-language',
}

# Extension Options
autodoc_default_options = {
    'imported-members': True,
    'member-order': 'groupwise',
    'members': True,
    'no-value': True,
    'show-inheritance': True,
    'undoc-members': True,
}
autodoc_mock_imports = [
    'federated_language.proto',
]
autodoc_type_aliases = {
    # TODO: b/388054994 - Document `federated_language` type aliases.
    # 'Array': 'federated_language.compiler.array.Array',
}
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}


def linkcode_resolve(domain, info):
  """Return the URL to source code corresponding to the Python object."""

  if domain != 'py':
    return None
  if not info['module']:
    return None
  if not info['fullname']:
    return None

  try:
    module = sys.modules[info['module']]
  except ImportError:
    return None

  attributes = info['fullname'].split('.')
  try:
    obj = functools.reduce(getattr, attributes, module)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  filename = os.path.relpath(filename, start=os.path.abspath('..'))
  if lineno != 0:
    lines = f'#L{lineno}-L{lineno + len(source)}'
  else:
    lines = ''
  return f'https://github.com/google-parfait/federated-language/blob/main/federated_language/{filename}{lines}'
