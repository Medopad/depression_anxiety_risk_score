"""Sphinx configuration."""
from datetime import datetime
import os
import sys

directory_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src/",
)
sys.path.insert(
    0,
    directory_path,
)

project = "Depression_Anxiety_Risk_Scores"
author = "BASTIEN ORSET"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
]


autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ["templates"]
