# tomopyui

Have you ever wondered to yourself one of the following:

- "I really don't want to learn a python API to reconstruct my tomography data" 
- "I really wish I knew what was going on during automatic tomography data alignment, and that it wasn't just a black box filled with math that gives me a bad result"
- "I really don't want to open another image stack in ImageJ"

`tomopyui` aims to provide a solution to these problems. Built on [tomopy](https://tomopy.readthedocs.io/en/latest/), [astra-toolbox](http://www.astra-toolbox.com/docs/install.html), [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/), and [mpl-interactions](https://mpl-interactions.readthedocs.io/en/stable/index.html), `tomopyui` is a graphical user interface (GUI) that will allow you to:

- Import tomography data
- Find the data's center of rotation (manually, or automatically from [tomopy](https://tomopy.readthedocs.io/en/latest/) 's centering algorithms)
- Iteratively align your data using [joint iterative reconstruction and reprojection](https://www.nature.com/articles/s41598-017-12141-9.pdf) and inspect the convergence at each iteration. 
- Look at your normalized/aligned/reconstructed data in the app, rather than pulling it up in ImageJ
- Try out all the reconstruction algorithms in a few clicks. Run them. Come back to folders filled with reconstructed data using all those algorithms. Some are better than others, and some are faster than others. 
- Process a dataset quickly to find standard values, save alignment and reconstruction metadata in JSON files for batch reconstruction later on. (still in development progress)

This application was developed at the Stanford Synchrotron Radiation Lightsource ([SSRL](https://www-ssrl.slac.stanford.edu/)) to aid in our alignment and reconstruction processes. It could certainly use _your_ help! See the {doc}`contributing` page for more information on how you can get involved.

## Usage

Open up [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) after [installing](#install), and run the following in the first cell:

```{jupyter-execute}
%matplotlib ipympl
import tomopyui.widgets.main as main

dashboard, file_import, center, prep, align, recon = main.create_dashboard()
dashboard
```

You can click through the tabs that you will see on the dashboard to check out the options available. Most of the features cannot be used on this webpage, but it gives you a flavor of what comes with the app.

Here is what it will look like when you open it on your computer:

```{image} _static/images/frontpage_usage.gif

```

There will be more examples of usage [TK](https://en.wikipedia.org/wiki/To_come_(publishing)) in the {doc}`examples` page.

This shows you how to upload your data and check it out using the [mpl-interactions](https://mpl-interactions.readthedocs.io/en/stable/index.html) interactive [hyperslicer](https://mpl-interactions.readthedocs.io/en/stable/examples/hyperslicer.html) and [histogram](https://mpl-interactions.readthedocs.io/en/stable/examples/hist.html) widgets.

## Install

:::{note}

If you are new to installing conda/pip packages, and/or you do not currently have a CUDA installation on your machine, see the {doc}`install` page for an in-depth (at least for Windows) guide.

:::

:::{note}

At the moment, this package only supports [tomopy](https://tomopy.readthedocs.io/en/latest/) and [astra-toolbox](http://www.astra-toolbox.com/docs/install.html) installed with CUDA. In the future, we will relax this requirement.  

:::

### Installing with CUDA
Once you are finished installing CUDA, navigate to the directory in which you would like to install tomopyui (use cd in the anaconda prompt to navigate):

```
cd your-install-directory-name
```

Clone the github repository:

```
git clone https://github.com/samwelborn/tomopyui.git
```

Navigate on into the tomopyui directory:

```
cd tomopyui
```

Run the following command:

```
conda env create -f environment.yml
```

This will install a new environment called tomopyui. To activate this environment:

```
conda activate tomopyui
```

Once you do that, you should see 

```
(tomopyui)
```

instead of (base) in your anaconda prompt. Finally, your last step is to install tomopyui. From the main directory (the one that has setup.py in it), run:

```
pip install .
```

### Installing without CUDA

Without CUDA, this program is useless for aligning/reconstructing tomography data. 

If you don't have CUDA and you just want to check out the ipywidgets, you can still do that using the environment.yml in the docs folder:

```
cd tomopyui
cd docs
conda env create -f environment.yml
```

Then, activate the environment and install:

```
conda activate tomopyui-docs
cd ..
pip install .
```

_Follow the links below for in-depth installation page and API._

```{toctree}
:maxdepth: 2

howitworks
install
examples
API <api/tomopyui>
contributing
```

```{toctree}
:caption: Examples
:maxdepth: 1
```