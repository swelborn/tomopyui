# TomoPyUI

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/front_page.mp4", autoplay=False)
```

Have you ever wondered to yourself one of the following:

- "I really don't want to learn a python API to reconstruct my tomography data" 
- "I really wish I knew what was going on during automatic tomography data alignment, and that it wasn't just a black box filled with math that gives me a bad result"
- "I really don't want to open another image stack in ImageJ"

`tomopyui` aims to provide a solution to these problems. Built on [tomopy](https://tomopy.readthedocs.io/en/latest/), [astra-toolbox](http://www.astra-toolbox.com/docs/install.html), [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/), [bqplot](), and [bqplot-image-gl](https://pypi.org/project/bqplot-image-gl/)  `tomopyui` is a highly interactive and reactive graphical user interface (GUI) that will allow you to:

- Import tomography data from various sources - raw beamline data (projections, references, and dark fields), or prenormalized data from another program (i.e., [TXM Wizard](https://sourceforge.net/projects/txm-wizard/))
- Find the data's center of rotation (manually, or automatically from [tomopy](https://tomopy.readthedocs.io/en/latest/) 's centering algorithms)
- Iteratively align your data using [joint iterative reconstruction and reprojection](https://www.nature.com/articles/s41598-017-12141-9.pdf) and inspect the convergence at each iteration. 
- Look at your normalized/aligned/reconstructed data in the app, rather than pulling it up in ImageJ
- Try out all the reconstruction algorithms in a few clicks. Run them. Come back to folders filled with reconstructed data using all those algorithms. Some are better than others.
- Process a dataset quickly to find standard values, save alignment and reconstruction metadata in JSON files for batch reconstruction later on.

At each part of this process, metadata about your data is saved so that you know what you did when you come back to it in a month or two or three or seven.

This application was developed at the Stanford Synchrotron Radiation Lightsource ([SSRL](https://www-ssrl.slac.stanford.edu/)) to aid in our alignment and reconstruction processes. It could certainly use _your_ help! See the {doc}`contributing` page for more information on how you can get involved.

## Usage

Open up [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) after [installing](#install), and run the following in the first cell:

```{jupyter-execute}
import tomopyui.widgets.main as main

dashboard_output, dashboard, file_import, prep, center, align, recon, dataexplorer = main.create_dashboard("SSRL_62C") # can be "SSRL_62C", "ALS_832", or "APS"
dashboard_output
```

You can click through the tabs that you will see on the dashboard to check out the options available. Most of the features cannot be used without a Jupyter environment, but it gives you a flavor of what comes with the app.

TomoPyUI is under development, and new features are being added with each commit. Video tutorials on usage can be found under the {doc}`examples` page -- the most recent walkthrough of its functionality is there, where SSRL 6-2C users from both [NREL](https://www.nrel.gov/) and [UCLA](https://www.chem.ucla.edu/scalar/) joined the TomoPyUI team for a tutorial. 

## Install

:::{note}

If you are new to installing conda/pip packages, and/or you do not currently have a CUDA installation on your machine, see the {doc}`install` page for an in-depth (at least for Windows) guide.

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

instead of (base) in your anaconda prompt. Finally, your last step is to install TomoPyUI. From the main directory (the one that has setup.py in it), run:

```
pip install -e .
```

### Installing without CUDA

Without CUDA, you will miss out on some of the features in TomoPyUI. You can still install it by doing the following

```
cd tomopyui
conda env create -f environment-nocuda.yml
```

Then, activate the environment and install:

```
conda activate tomopyui
cd ..
pip install -e .
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
