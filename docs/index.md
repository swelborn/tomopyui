
# tomopyui's Documentation

A graphical user interface for tomopy.


## Usage

Open up Jupyter Lab, and run the following in the first cell:

```{jupyter-execute}
%matplotlib ipympl
import tomopyui.widgets.main as main

dashboard, file_import, center, prep, align, recon = main.create_dashboard()
dashboard
```

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

Run the following command :

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

If you don't have CUDA or and just want to check out the ipywidgets, you can still do that using the environment.yml in the docs folder:

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

install
API <api/tomopyui>
```

```{toctree}
:caption: Examples
:maxdepth: 1
```
