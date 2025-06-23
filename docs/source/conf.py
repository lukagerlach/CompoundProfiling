# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Compound Profiling"
copyright = "2025, Vincent von Häfen, Luka Gerlach"
author = "Vincent von Häfen, Luka Gerlach"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]  # Commented out - no custom static files needed

# PyData theme options
html_theme_options = {
    "logo": {
        "text": "Compound Profiling",
    },
    "show_toc_level": 2,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# MyST parser settings
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
]

# Tell MyST to treat .md files without extension as text
source_suffix = {
    '.rst': None,
    '.md': None,
}
