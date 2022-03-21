# How it works

## Packages tomopyui relies on

As mentioned on the [home page](https://tomopyui.readthedocs.io/en/latest/), `tomopyui` is built on many open-source python packages including:

**tomopy**
- [Documentation](https://tomopy.readthedocs.io/en/latest/)
- Automatic center of rotation finding using [`tomopy.recon.rotation.find_center`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) and [`tomopy.recon.rotation.find_center_vo`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html)
- Manual center of rotation finding using [`tomopy.recon.rotation.write_center`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html)
- Iterative alignment algorithm
- CPU-based reconstruction algorithms ([TK](https://en.wikipedia.org/wiki/To_come_(publishing)))
- Wrapper for ASTRA algorithms

**astra-toolbox**
- [Documentation](https://www.astra-toolbox.com/)
- Tomographic reconstruction toolbox with many GPU-accelerated algorithms

**ipywidgets**
- [Documentation](https://ipywidgets.readthedocs.io/en/latest/)
- The user interface is created inside of [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) using ipywidgets.

**cupy**
- [Documentation](https://docs.cupy.dev/en/stable/overview.html)
- Sends [numpy](https://numpy.org/doc/1.21/) arrays to the GPU for calculations 

**tomocupy**
- [Documentation TK](https://en.wikipedia.org/wiki/To_come_(publishing))
- Utilizes power of GPU-acceleration to speed up automatic alignment
- Built from [tomopy](https://tomopy.readthedocs.io/en/latest/), but sends data to the GPU to do some of the calculations.
- Only a few functions available now, but could be expanded significantly 
- Included as a module in tomopyui (for now)
- Helps run a lot of the backend of tomopyui


## Code structure

The code is divided into the frontend (currently <!-- {doc}`api/tomopyui.widgets` -->) and the backend (currently <!-- {doc}`api/tomopyui.backend` --> and <!-- {doc}`api/tomopyui.tomocupy` -->). This could be divided differently/renamed later, but it is a starting point.

### Frontend

When the user calls `tomopyui.widgets.main.create_dashboard`, they are creating several different objects:

<!-- {module}`~tomopyui.widgets.imports` -->
- Helps import data. 
- Creates the first tab of the dashboard, which contains:
    1. [ipyfilechooser](https://github.com/crahan/ipyfilechooser) widget for choosing a file.
    2. [FloatText](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#FloatText) widgets for start and end angle.
    3. [IntText](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#IntText) widget for number of angles. This currently does not do anything: the file import automatically grabs the number of angles.
- It is a starting point to many of the other classes 
    - Other classes use file path information from <!-- {class}`~tomopyui.widgets.meta.Import` --> to import a <!-- {class}`~tomopyui.backend.tomodata.TomoData` --> object and run the algorithms on the backend.

<!-- {module}`~tomopyui.widgets.view` -->
- Helps plot data
- Uses [hyperslicer](https://mpl-interactions.readthedocs.io/en/stable/examples/hyperslicer.html) and [histogram](https://mpl-interactions.readthedocs.io/en/stable/examples/hist.html) to make this interactive in Jupyter.

<!-- {module}`~tomopyui.widgets.center` -->
- Creates Center tab in the dashboard
- Contains two [Accordion](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Accordion) widgets:
    1. Button clicks use [`tomopy.recon.rotation.find_center`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) and [`tomopy.recon.rotation.find_center_vo`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) for automatic center finding.
    2. Button clicks use <!-- {doc}`api/tomopyui.backend.util.center` -->.write_center, which is a copy of [`tomopy.recon.rotation.write_center`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) to reconstruct a given slice of the data at various centers. The user can then maneuver the slider to find the best center by eye.

<!-- {module}`~tomopyui.widgets.analysis` -->
- Creates Align tab in the dashboard
- Contains a [RadioButton](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#RadioButtons) widgets to activate alignment, or to use a full or partial dataset.
- Contains several [Accordion](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Accordion) widgets:
    1. Plot Projection Images:
        - Uses <!-- {class}`~tomopyui.widgets.meta.Plotter` --> class to plot projection images before alignment. One can click the "set projection range" button after selecting a window (see [this gif](./index.md#usage)), and it will set the projection range to the current range on the hyperslicer plot.
        - Can save the animation as an mp4 using a button. [TK](https://en.wikipedia.org/wiki/To_come_(publishing)) 
    2. Methods:
        - Selection of various CUDA-based reconstruction algorithms for the alignment. Selecting more than one of these will perform the alignment multiple times (must be fixed in backend, first).
    3. Save Options:
        - Various save options, documented in <!-- {class}`~tomopyui.widgets.meta.Align` -->
    4. Alignment options:
        - Various alignment options, documented in <!-- {class}`~tomopyui.widgets.meta.Align` -->
<!-- 
{class}`~tomopyui.widgets.analyis`
- Subclass of {class}`~tomopyui.widgets.meta.Align`. Variations include what metadata to set, and some buttons/checkboxes.  -->

### Backend

[Documentation TK](https://en.wikipedia.org/wiki/To_come_(publishing))