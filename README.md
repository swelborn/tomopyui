# tomopyui

[![License](https://img.shields.io/pypi/l/tomopyui.svg?color=green)](https://github.com/samwelborn/tomopyui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tomopyui.svg?color=green)](https://pypi.org/project/tomopyui)
[![Python Version](https://img.shields.io/pypi/pyversions/tomopyui.svg?color=green)](https://python.org)
[![CI](https://github.com/samwelborn/tomopyui/actions/workflows/ci/badge.svg)](https://github.com/samwelborn/tomopyui/actions)
[![codecov](https://codecov.io/gh/samwelborn/tomopyui/branch/master/graph/badge.svg)](https://codecov.io/gh/samwelborn/tomopyui)

## Install

Create a fresh environment so you don't break anything else:

```
conda create -n tomopyui
conda activate tomopyui
```

Then, install run the following commands in order:

```
conda install -c conda-forge python=3.9 tomopy dxchange jupyterlab astropy
conda install -c conda-forge/label/jupyterlab_widgets_rc -c conda-forge/label/ipywidgets_dev -c conda-forge/label/widgetsnbextension_dev ipywidgets widgetsnbextension jupyterlab_widgets
conda install -c conda-forge ipyfilechooser
mamba install -c conda-forge ipympl --no-deps
pip install mpl-interactions --no-deps
```

After creating this environment, navigate to the directory you downloaded
this repository to, and run the following command:

```
pip install .
```

or 

```
pip install -e .
```

for helping to develop this GUI.

