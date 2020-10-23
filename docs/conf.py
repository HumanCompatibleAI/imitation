# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import imitation  # pytype: disable=import-error

project = "imitation"
copyright = "2019, Center for Human-Compatible AI"  # noqa: A001
author = "Center for Human-Compatible AI"

# The full version, including alpha/beta/rc tags
release = imitation.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Customization -----------------------------------------------------------


def no_namedtuple_attrib_docstring(app, what, name, obj, options, lines):
    """Remove redundant documentation in named tuples.

    Worksaround https://github.com/sphinx-doc/sphinx/issues/7353 -- adapted from
    https://chrisdown.name/2015/09/20/removing-namedtuple-docstrings-from-sphinx.html
    """
    is_namedtuple_docstring = 1 <= len(lines) <= 2 and lines[0].startswith(
        "Alias for field number"
    )
    if is_namedtuple_docstring:
        # We don't return, so we need to purge in-place
        del lines[:]


def setup(app):
    app.connect(
        "autodoc-process-docstring",
        no_namedtuple_attrib_docstring,
    )
