import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'NODIS'
author = 'Roland V. Bumbuc'
release = '0.1.0'
copyright = '2026, Roland V. Bumbuc'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build']
html_theme = 'sphinx_rtd_theme'
html_static_path = []

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
