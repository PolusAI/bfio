# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'bfio'
author = 'Nicholas J Schaub'

# The full version, including alpha/beta/rc tags
with open(str(Path(__file__).parent.parent.parent.joinpath('VERSION')),'r') as fr:
    release = fr.readline()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints'
]

napoleon_use_param = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['..']

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# Shows both the class docstring and __init__
autoclass_content = 'both'

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Mock imports to autobuild
autodoc_mock_imports = ["javabridge",
                        "bioformats"]

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

autodoc_default_options = {
    'members': True,
    'special-members': '__getitem__,__call__,__setitem__'
    # 'exclude-members': '__weakref__'
}

# Set the master doc
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']