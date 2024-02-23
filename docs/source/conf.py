# Configuration file for the Sphinx documentation builder.

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
# Attempt to import tomopyui version, else default to "unknown"
_release = {}
exec(
    compile(
        open("../../tomopyui/_version.py").read(), "../../tomopyui/_version.py", "exec"
    ),
    _release,
)
v = Version(_release["__version__"])
version = f"{v.major}.{v.minor}"
release = _release["__version__"]

# -- Project information -----------------------------------------------------

master_doc = "index"
project = "tomopyui"
copyright = "2024, Samuel Scott Welborn"
author = "Samuel Scott Welborn"
language = "en"

# -- Sphinx Extensions and Configuration -------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

# Intersphinx mapping to link to the documentation of other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/latest/", None),
    "tomopy": ("https://tomopy.readthedocs.io/en/latest/", None),
}

# MyST-NB settings for Jupyter notebook documentation
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]

nb_execution_mode = "cache"

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
htmlhelp_basename = "tomopyuidoc"
html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/tomopyui",
            "icon": "fa-solid fa-box",
        }
    ],
    "use_edit_page_button": True,
    "github_url": "https://github.com/swelborn/tomopyui",
    "navbar_end": [
        # disabled until widget dark variables are available
        # "theme-switcher",
        "navbar-icon-links",
    ],
}


# html_theme_options = {
#     "repository_url": "https://github.com/samwelborn/tomopyui",
#     "use_edit_page_button": True,
#     "use_repository_button": True,
#     "use_issues_button": True,
#     "path_to_docs": "docs",
# }

html_context = {
    # disabled until widget dark variables are available
    "default_mode": "light",
    "doc_path": "docs",
    "github_repo": "tomopyui",
    "github_user": "swelborn",
    "github_version": "main",
}

html_logo = "_static/images/logo.png"
html_favicon = "_static/images/tomopy-favicon.svg"
html_title = "tomopyui"
templates_path = ["_templates"]


# Define any other necessary functions
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
