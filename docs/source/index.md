# TomoPyUI

<video controls width="100%">
  <source src="_static/videos/front_page.mp4" type="video/mp4">
</video>

Have you ever wondered to yourself one of the following:

- "I really don't want to learn a python API to reconstruct my tomography data"
- "I really wish I knew what was going on during automatic tomography data alignment, and that it wasn't just a black box filled with math that gives me a bad result"
- "I really don't want to open another image stack in ImageJ"

`tomopyui` aims to provide a solution to these problems. Built on [tomopy](https://tomopy.readthedocs.io/en/latest/), [astra-toolbox](http://www.astra-toolbox.com/docs/install.html), [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/), [bqplot](https://bqplot.github.io/bqplot/), and [bqplot-image-gl](https://pypi.org/project/bqplot-image-gl/)  `tomopyui` is a highly interactive and reactive graphical user interface (GUI) that will allow you to:

- Import tomography data from various sources - raw beamline data (projections, references, and dark fields), or prenormalized data from another program (i.e., [TXM Wizard](https://sourceforge.net/projects/txm-wizard/))
- Find the data's center of rotation (manually, or automatically from [tomopy](https://tomopy.readthedocs.io/en/latest/)'s centering algorithms)
- Iteratively align your data using [joint iterative reconstruction and reprojection](https://www.nature.com/articles/s41598-017-12141-9.pdf) and inspect the convergence at each iteration.
- Look at your normalized/aligned/reconstructed data in the app, rather than pulling it up in ImageJ
- Try out all the reconstruction algorithms in a few clicks. Run them. Come back to folders filled with reconstructed data using all those algorithms. Some are better than others.
- Process a dataset quickly to find standard values, save alignment and reconstruction metadata in JSON files for batch reconstruction later on.

At each part of this process, metadata about your data is saved so that you know what you did when you come back to it in a month or two or three or seven.

This application was developed at the Stanford Synchrotron Radiation Lightsource ([SSRL](https://www-ssrl.slac.stanford.edu/)) to aid in our alignment and reconstruction processes. See the {doc}`contributing` page for more information on how you can get involved.

```{toctree}
:maxdepth: 2

install
usage
how-to
howitworks
contributing
```
