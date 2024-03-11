# Documentation

## Sphinx
This folder contains all the documentation generated with the sphinx documentation tool

Sphinx is an autmatic documentation generator that generates documentation for python by 
reading in files, the result is an html page

### Setup
pip install -U sphinx

#### Building:
sphinx-apidoc -o docs .
cd docs
sphinx-quickstart
(add "modules" in index.rst)
(add the right paths in conf.py)
make html
cd _build/html
python3 -m http.server 8000

### Additional things to try:
#### Change theme
pip install sphinx-rtd-theme
(Then change html theme in conf.py)

#### Add extensions
Add extensions in conf.py, such as viewing source code etc.

#### Clean previous build
make clean

#### Update makefile
(change desired output directory)