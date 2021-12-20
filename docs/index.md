# tomopyui: user-focused tomographic reconstruction

`tomopyui` is a graphical user interface for [`tomopy`](https://tomopy.readthedocs.io/en/latest/)),
an open-source Python package for tomographic data processing and reconstruction. 
Using many functions derived from tomopy, tomopyui hopes to offer a more user-friendly solution for the typical synchrotron TXM user who does not have time to learn python, or to learn a new python package. You can improve your tomographic reconstruction workflow by:

- Importing data directly in the UI - no need to specify an exact path for your dataset
- Find your center of rotation quickly using both automatic and manual center-finding methods
- Avoiding constant window-switching to ImageJ to check your data before alignment or reconstruction.
- Seeing your alignment take place in front of you.
- Try many different reconstruction algorithms without changing so many options.
- Preprocessing one set of data for a 'standard' alignment and reconstruction routine, which can then be applied to multiple sets of data.

## Installation

`tomopyui` has been developed on recent (v8.0.0, as of 12/2021) versions of [`ipywidgets`](https://ipywidgets.readthedocs.io/en/latest/) and its dependencies. [`mpl-interactions`](https://mpl-interactions.readthedocs.io/en/stable/index.html) works (as far as I can tell) ontop of `ipywidgets` v8, but has not yet relaxed its requirements. 

To install, you have to force-install mpl-interactions after installing the other packages. 

First, install conda from [here](https://www.anaconda.com/) and read up on it if you are not familiar.

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
conda install -c conda-forge ipympl --no-deps
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

for helping to develop the GUI.

In order to create the documentation, you will have to install sphinx:

```
conda install sphinx
conda install -c conda-forge myst-nb
mamba install -c conda-forge jupyter_sphinx
```

## Basic usage

To create a dashboard, the only cell you have to run is the following (note, I 
have only used this in Jupyter Lab):

```python
# if running this code in a Jupter notbeook or JupyterLab
%reload_ext autoreload
%autoreload 2
%matplotlib ipympl
import tomopyui.widgets.main as main

dashboard, file_import, center, prep, align, recon = main.create_dashboard()
dashboard
```
**If you are in a Jupyter Notebook the output should look like this:**

```{image} _static/images/centering.gif

```

In addition to [`tomopy`](https://tomopy.readthedocs.io/en/latest/)), `tomopyui` makes heavy use of [`ipywidgets`](https://ipywidgets.readthedocs.io/en/latest/) and [`mpl-interactions`](https://mpl-interactions.readthedocs.io/en/stable/index.html).


## Getting Help

If you have a question on using this dashboard, you can raise an issue [here](https://github.com/samwelborn/tomopyui/issues).

_Follow the links below for further information on installation, functions, and plot examples._

```{toctree}
:maxdepth: 3

API <api/tomopyui>

contributing
```
<!-- install -->
<!-- gallery/index -->
<!-- ```{toctree}
:caption: Tutorials
:maxdepth: 1

examples/usage.ipynb
examples/context.ipynb
examples/mpl-sliders.ipynb
examples/custom-callbacks.ipynb
examples/animations.ipynb
examples/range-sliders.ipynb
examples/scalar-arguments.ipynb
examples/tidbits.md
``` -->

<!-- ```{toctree}
:caption: Specific Functions
:maxdepth: 1

examples/hyperslicer.ipynb
examples/plot.ipynb
examples/scatter.ipynb
examples/imshow.ipynb
examples/hist.ipynb
examples/scatter-selector.ipynb
examples/image-segmentation.ipynb
examples/zoom-factory.ipynb
examples/heatmap-slicer.ipynb
``` -->

<!-- ```{toctree}
:caption: Showcase
:maxdepth: 1

examples/lotka-volterra.ipynb
examples/rossler-attractor.ipynb
``` -->

### Modules

- {ref}`modindex`