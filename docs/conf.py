# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

project = "torch_simple_timing"
copyright = "2023, Victor Schmidt"
author = "Victor Schmidt"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

release = (
    [
        line.split("=")[-1].strip()
        for line in (Path(__file__).resolve().parent.parent / "pyproject.toml")
        .read_text()
        .splitlines()
        if line.startswith("version")
    ][0]
    .replace("'", "")
    .replace('"', "")
)

extensions = [
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Options for autoapi -----------------------------------------------------

autodoc_typehints = "description"
autoapi_type = "python"
autoapi_dirs = ["../torch_simple_timing/"]
autoapi_member_order = "alphabetical"
autoapi_template_dir = "_autoapi_templates"
autoapi_python_class_content = "init"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",
    "special-members",
]
autoapi_keep_files = False

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
mathjax3_config = {
    "tex": {
        "inlineMath": [
            ["$", "$"],
            ["\\(", "\\)"],
        ],
        "processEscapes": True,
    },
}


html_theme = "furo"


# https://github.com/tox-dev/sphinx-autodoc-typehints
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_defaults = "comma"
