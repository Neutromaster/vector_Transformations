# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Transformations'
copyright = '2026, Vishnu Kumar'
author = 'Vishnu Kumar'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  
'sphinx.ext.autodoc',  
'sphinx.ext.napoleon', # supports Google/Numpy docstrings  
'sphinx.ext.mathjax', # renders equations  
'sphinx.ext.viewcode', # show source code  
'myst_parser', # markdown support  
'sphinx_copybutton'  
]



html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
