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

**bqplot**

- [Documentation](https://bqplot.readthedocs.io/en/latest/)

**bqplot-image-gl**

- [pip](https://pypi.org/project/bqplot-image-gl/)

## Structure

The code is divided into the frontend (`tomopyui.widgets`) and the backend (`tomopyui.backend` and `tomopyui.tomocupy`).

When the user calls `tomopyui.widgets.main.create_dashboard`, they are creating several different objects:

1. `Import` tab:
    - The type of `Import` tab is dictated by the user's choice in string on the call from the Jupyter notebook (e.g., "SSRL_62C" or "ALS_832")
    - Holds both raw and prenormalized `Uploader`s, which hold a specific type of `RawProjections` (depending on the choice above) and a general `Prenormalized_Projections` object, respectively.
    - `Uploader`s upload the data (hold buttons, viewers, etc.), and `Projection`s hold the data.
    - `Projection` objects also hold specific types of metadata. Storing metadata along the processing pipeline is critical.

2. `Center` tab:
    - Used to interactively find the center of rotation
    - Contains two [Accordion](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Accordion) widgets:
        1. [`tomopy.recon.rotation.find_center`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) and [`tomopy.recon.rotation.find_center_vo`](https://tomopy.readthedocs.io/en/latest/api/tomopy.recon.rotation.html) for automatic center finding.
        2. Reconstruction of a given slice of the data at various centers. The user can then maneuver the slider to find the best center by eye.

3. `Align` tab:
    - Used to align jittery data using the automatic alignment algorithms built into tomopy.
    - Align only a part of a dataset by selecting an ROI in the "Imported Projections" viewer.

4. `Recon` tab:
    - Used to reconstruct data using one of many algorithms available from TomoPy and Astra Toolbox.

5. `Data Explorer` tab:
    - Used to look at prior alignments.

All of these objects talk to `view.py`, which holds the image viewers.
