#!/usr/bin/env python

from ipywidgets import *
from ._import import import_helpers
from ._shared import helpers
from ._shared._init_widgets import init_widgets, _set_widgets_from_load_metadata
from ipyfilechooser import FileChooser
from mpl_interactions import (
    hyperslicer,
    ioff,
    interactive_hist,
    zoom_factory,
    panhandler,
)
from bqplot_image_gl import ImageGL
from tomopyui.backend.util.center import write_center
from tomopy.recon.rotation import find_center_vo, find_center, find_center_pc
from tomopyui.backend.util.metadata_io import (
    save_metadata,
    load_metadata,
    metadata_to_DataFrame,
)

# includes astra_cuda_recon_algorithm_kwargs, tomopy_recon_algorithm_kwargs,
# and tomopy_filter_names, extend_description_style
from tomopyui._sharedvars import *
from tomopyui.backend.tomodata import TomoData

import functools
import os
import tomopy.prep.normalize
import tomopyui.backend.tomodata as td
import matplotlib.pyplot as plt
import numpy as np
import logging
import bqplot as bq
import pathlib


class Import:
    """
    Class to import tomography data. At this point, it is assumed that the data
    coming in is already normalized externally.

    Attributes
    ----------
    angle_start, angle_end : double
        Start and end angles of the data being imported
    num_theta : int
        Number of theta values in the dataset. Currently does not do anything
        in the backend.
    prj_shape : tuple, (Z, Y, X)
        Shape pulled from image imported. After choosing a file in the
        Filechooser object, this will update. This is sent to Align and Recon
        objects to instantiate sliders.
    angles_textboxes : list of :doc:`Textbox <ipywidgets:index>`
        List of :doc:`Textbox <ipywidgets:index>` widgets for angles.
    import_opts_list : list of str
        Right now, the only option on import is "rotate". If other options were
        to be added, you can add to this list. Each of these options will
        create a :doc:`Checkbox <ipywidgets:index>` in self.opts_checkboxes.
    opts_checkboxes : list of :doc:`Checkbox <ipywidgets:index>`
        Created from import_opts_list.
    fpath : str
        Updated when file or filepath is chosen.
    fname : str
        Updated when file is chosen (if only directory is chosen, fname="")
    ftype : "npy", "tiff"
        Filetypes. Should be expanded to h5 files in future versions.
    filechooser : FileChooser()
        Filechooser widget
    prj_range_x : (0, number of x pixels)
        Retrieved from reading image metadata after choosing a file or folder.
    prj_range_y : (0, number of y pixels)
        Retrieved from reading image metadata after choosing a file or folder.
    prj_range_z : (0, number of z pixels)
        Not currently working
    tomo : tomodata.TomoData()
        Created after calling make_tomo.
    wd : str
        Current working directory, set after choosing file or folder. This is
        defaulted to fpath
    log : logging.logger
        Logger used throughout the program
    log_handler :
        Can use this as context manager, see
        :doc:`ipywidgets docs <ipywidgets:index>`.
    metadata : dict
        Keys are attributes of the class. Values are their values. They are
        updated by `Import.set_metadata()`.
    tab : :doc:`HBox <ipywidgets:index>`
        Box for all the :doc:`widgets <ipywidgets:index>` that are in the
        `Import` tab. Created at end of
        __init__().

    """

    def __init__(self):

        # Init textboxes
        self.angle_start = -90.0
        self.angle_end = 90.0
        self.num_theta = 360
        self.prj_shape = None
        self.prj_range_x = (0, 100)
        self.prj_range_y = (0, 100)
        self.prj_range_z = (0, 100)
        self.tomo = None
        self.angles_textboxes = import_helpers.create_angles_textboxes(self)

        # Init checkboxes
        self.import_opts_list = ["rotate"]
        self.import_opts = {key: False for key in self.import_opts_list}
        self.opts_checkboxes = helpers.create_checkboxes_from_opt_list(
            self.import_opts_list, self.import_opts, self
        )

        # Init filechooser
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_file_information)
        self.filechooser.title = "Import Normalized Tomogram:"
        self.wd = None

        # Init filechooser for align metadata
        self.fpath_align = None
        self.fname_align = None
        self.filechooser_align = FileChooser()
        self.filechooser_align.register_callback(self.update_file_information_align)
        self.filechooser_align.title = "Import Alignment Metadata:"

        # Init filechooser for recon metadata
        self.fpath_recon = None
        self.fname_recon = None
        self.filechooser_recon = FileChooser()
        self.filechooser_recon.register_callback(self.update_file_information_recon)
        self.filechooser_recon.title = "Import Reconstruction Metadata:"

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

        # Init metadata
        self.metadata = {}
        self.set_metadata()

        # Create tab
        self.make_tab()

    def set_wd(self, wd):
        """
        Sets the current working directory of `Import` class and changes the
        current directory to it.
        """
        self.wd = wd
        os.chdir(wd)

    def set_metadata(self):
        """
        Sets relevant metadata for `Import`
        """
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta,
            "prj_range_x": self.prj_range_x,
            "prj_range_y": self.prj_range_y,
        } | self.import_opts

    def update_file_information(self):
        """
        Callback for `Import`.filechooser.
        """
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_wd(self.fpath)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.set_metadata()
        self.get_prj_shape()
        self.set_prj_ranges()
        self.set_metadata()

    def update_file_information_align(self):
        """
        Callback for filechooser_align.
        """
        self.fpath_align = self.filechooser_align.selected_path
        self.fname_align = self.filechooser_align.selected_filename
        self.set_metadata()

    def update_file_information_recon(self):
        """
        Callback for filechooser_recon.
        """
        self.fpath_recon = self.filechooser_recon.selected_path
        self.fname_recon = self.filechooser_recon.selected_filename
        self.set_metadata()

    def get_prj_shape(self):
        """
        Grabs the image shape depending on the filename. Does this without
        loading the image into memory.
        """
        if self.fname.__contains__(".tif"):
            self.prj_shape = helpers.get_img_shape(
                self.fpath,
                self.fname,
                "tiff",
                self.metadata,
            )
        elif self.fname.__contains__(".npy"):
            self.prj_shape = helpers.get_img_shape(
                self.fpath,
                self.fname,
                "npy",
                self.metadata,
            )
        elif self.fname == "":
            self.prj_shape = helpers.get_img_shape(
                self.fpath, self.fname, "tiff", self.metadata, folder_import=True
            )

    def set_prj_ranges(self):
        self.prj_range_x = (0, self.prj_shape[2] - 1)
        self.prj_range_y = (0, self.prj_shape[1] - 1)
        self.prj_range_z = (0, self.prj_shape[0] - 1)

    def make_tab(self):
        """
        Creates the HBox which stores widgets.
        """
        import_widgets = [
            [self.filechooser],
            self.angles_textboxes,
            self.opts_checkboxes,
        ]
        import_widgets = [item for sublist in import_widgets for item in sublist]
        self.tab = VBox(
            [
                HBox(import_widgets),
                HBox([self.filechooser_align, self.filechooser_recon]),
            ]
        )

    def make_tomo(self):
        """
        Creates a `~tomopyui.backend.tomodata.TomoData` object and stores it in
        `Import`.

        .. code-block:: python

            # In Jupyter:

            # Cell 1:
            from ipywidgets import *
            from tomopyui.widgets.meta import Import
            from
            a = Import()
            a.tab
            # You should see the HBox widget, you can select your file.

            # Cell2:
            a.make_tomo() # creates tomo.TomoData based on inputs
            a.tomo.prj_imgs # access the projections like so.

        """
        self.tomo = td.TomoData(metadata=self.metadata)


class Plotter:
    """
    Class for plotting. Creates
    :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` for `Center`,
    `Align`, and `Recon` classes.

    Attributes
    ----------
    Import : `Import`
        Needs an import object to be constructed.
    DataExplorer : `DataExplorer`
        Optionally imports a `DataExplorer` object
    prj_range_x_slider : :doc:`IntRangeSlider <ipywidgets:index>`
        Used in both Align and Recon as their x range slider.
    prj_range_y_slider : :doc:`IntRangeSlider <ipywidgets:index>`
        Used in both Align and Recon as their y range slider.
    set_range_button : :doc:`Button <ipywidgets:index>`
        Used in both Align and Recon for setting the range to the
        current plot range on the image shown in the
        :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` (left fig).
    slicer_with_hist_fig : :doc:`matplotlib subplots <matplotlib:api/_as_gen/matplotlib.pyplot.subplots>`
        Figure containing an :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` (left) and
        a :doc:`histogram <mpl-interactions:examples/hist>` (right) associated with that slice.
    threshold_control : :class:`mpl-interactions:mpl_interactions.controller.Controls`
        Comes from :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>`. Contains slider associated
        with the current slice. See mpl-interactions for details.
    threshold_control_list : list of :doc:`ipywidgets <ipywidgets:index>`
        Allows for organization of the widgets after creating the :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>`.
    save_animation_button : :doc:`Button <ipywidgets:index>`
        Not implemented in `Align` or `Recon` yet (TODO). Enables saving of mp4
        of the :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` with its current :doc:`histogram <mpl-interactions:examples/hist>` threshold range.
    """

    def __init__(self, Import=None, DataExplorer=None):

        self.DataExplorer = DataExplorer
        self.Import = Import
        self._init_widgets()

    def _init_widgets(self):
        self.prj_range_x_slider = IntRangeSlider(
            value=[0, 10],
            min=0,
            max=10,
            step=1,
            description="Projection X Range:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=extend_description_style,
        )
        self.prj_range_y_slider = IntRangeSlider(
            value=[0, 10],
            min=0,
            max=10,
            step=1,
            description="Projection Y Range:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=extend_description_style,
        )
        self.set_range_button = Button(
            description="Click to set current range to plot range.",
            layout=Layout(width="auto"),
        )
        self.slicer_with_hist_fig = None
        self.threshold_control = None
        self.threshold_control_list = None
        self.save_animation_button = None

    def create_slicer_with_hist(self, plot_type="prj", imagestack=None, Center=None):
        """
        Creates a plot with a :doc:`histogram <mpl-interactions:examples/hist>`
        for a given set of data. Sets Plotter attributes: slicer_with_hist_fig,
        threshold_control, threshold_control_list, and set_range_button.

        Parameters
        -----------
        plot_type : "prj" or "center"
            Choice will determine what the :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` will show (projection
            images or reconstructed data with different centers, respectively).
        imagestack : 3D `numpy.ndarray`
            Images to show in :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>`
        Center : `Center`
            Used to title the plot with its attribute index_to_try.

        """

        # Turn off immediate display of plot.
        with plt.ioff():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout="tight")
        fig.suptitle("")
        fig.canvas.header_visible = False

        # check what kind of plot it is
        if plot_type == "prj":
            self.Import.make_tomo()
            imagestack = self.Import.tomo.prj_imgs
            theta = self.Import.tomo.theta
            slider_linsp = theta * 180 / np.pi
            slider_str = "Image No:"
            ax1.set_title("Projection Images")
        if plot_type == "center":
            imagestack = imagestack
            slider_linsp = (0, imagestack.shape[0])
            slider_str = "Center Number:"
            ax1.set_title(f"Reconstruction on slice {Center.index_to_try}")

        ax2.set_title("Image Intensity Histogram")
        ax2.set_yscale("log")

        # updates histogram based on slider, z-axis of hyperstack
        def histogram_data_update(**kwargs):
            return imagestack[threshold_control.params[slider_str]]

        def histogram_lim_update(xlim, ylim):
            current_ylim = [ylim[0], ylim[1]]
            ax2.set_ylim(ylim[0], ylim[1])
            current_xlim = [xlim[0], xlim[1]]
            ax2.set_xlim(xlim[0], xlim[1])

        # creating slicer, thresholding is in vmin_vmax param.
        # tuples here create a slider in that range
        threshold_control = hyperslicer(
            imagestack,
            vmin_vmax=("r", imagestack.min(), imagestack.max()),
            play_buttons=True,
            play_button_pos="right",
            ax=ax1,
            axis0=slider_linsp,
            names=(slider_str,),
        )
        current_xlim = [100, -100]
        current_ylim = [1, -100]
        for i in range(imagestack.shape[0]):
            image_histogram_temp = np.histogram(imagestack[i], bins=100)
            if image_histogram_temp[1].min() < current_xlim[0]:
                current_xlim[0] = image_histogram_temp[1].min()
            if image_histogram_temp[1].max() > current_xlim[1]:
                current_xlim[1] = image_histogram_temp[1].max()
            if image_histogram_temp[0].max() > current_ylim[1]:
                current_ylim[1] = image_histogram_temp[0].max()

        image_histogram = interactive_hist(
            histogram_data_update,
            xlim=("r", current_xlim[0], current_xlim[1]),
            ylim=("r", 1, current_ylim[1]),
            bins=100,
            ax=ax2,
            controls=threshold_control[slider_str],
            use_ipywidgets=True,
        )

        # registering to change limits with sliders
        image_histogram.register_callback(
            histogram_lim_update, ["xlim", "ylim"], eager=True
        )

        # finding lowest pixel value on image to set the lowest scale to that
        vmin = imagestack.min()
        vmax = imagestack.max()

        # putting vertical lines
        lower_limit_line = ax2.axvline(vmin, color="k")
        upper_limit_line = ax2.axvline(vmax, color="k")

        # putting vertical lines
        def hist_vbar_range_callback(vmin, vmax):
            lower_limit_line.set_xdata(vmin)
            upper_limit_line.set_xdata(vmax)

        threshold_control.register_callback(
            hist_vbar_range_callback, ["vmin", "vmax"], eager=True
        )

        # allowing zoom/pan on scroll/right mouse drag
        self.pan_handler = panhandler(fig)
        disconnect_zoom = zoom_factory(ax1)
        disconnect_zoom2 = zoom_factory(ax2)

        # sets limits according to current image limits
        def set_img_lims_on_click(click):
            xlim = [int(np.rint(x)) for x in ax1.get_xlim()]
            ylim = [int(np.rint(x)) for x in ax1.get_ylim()]
            if xlim[0] < 0:
                xlim[0] = 0
            if xlim[1] > imagestack.shape[2]:
                xlim[1] = imagestack.shape[2]
            if ylim[1] < 0:
                ylim[1] = 0
            if ylim[0] > imagestack.shape[1]:
                ylim[0] = imagestack.shape[1]

            xlim = tuple(xlim)
            ylim = tuple(ylim)
            self.prj_range_x_slider.value = xlim
            self.prj_range_y_slider.value = ylim[::-1]
            self.set_range_button.button_style = "success"
            self.set_range_button.icon = "square-check"

        self.set_range_button.on_click(set_img_lims_on_click)

        # saving some things in the object
        self.slicer_with_hist_fig = fig
        self.threshold_control = threshold_control
        self.threshold_control_list = [
            slider for slider in threshold_control.vbox.children
        ]

    def _swap_axes_on_click(self, imagestack, image, slider):
        imagestack = np.swapaxes(imagestack, 0, 1)
        slider.max = imagestack.shape[0] - 1
        slider.value = 0
        image.image = imagestack[0]
        return imagestack

    def _remove_high_low_intensity_on_click(self, imagestack, scale, slider):
        vmin, vmax = np.percentile(imagestack, q=(0.5, 99.5))
        slider.min = vmin
        slider.max = vmax
        self._set_bqplot_hist_range(scale, vmin, vmax)

    def _set_bqplot_hist_range(self, scale, vmin, vmax):
        scale["image"].min = vmin
        scale["image"].max = vmax

    def _create_one_plot(self, imagestack, title):
        fig, plotted_image, scale_image = self._create_bqplot_from_imagestack(
            imagestack[0], title[0]
        )
        figs = [fig]
        images = [plotted_image]
        scales = [scale_image]

        def change_image(change):
            plotted_image.image = imagestack[change.new]

        slider = IntSlider(
            value=0,
            min=0,
            max=imagestack.shape[0] - 1,
            step=1,
        )
        slider.observe(change_image, names="value")
        sliders = [slider]

        play = Play(
            value=0,
            min=0,
            max=imagestack.shape[0] - 1,
            step=1,
            interval=100,
            disabled=False,
        )
        jslink((play, "value"), (slider, "value"))
        plays = [play]

        return figs, images, scales, sliders, plays

    def _set_bqplot_hist_range(self, scale, vmin, vmax):
        scale["image"].min = vmin
        scale["image"].max = vmax

    def _create_two_plots_with_two_sliders(self, imagestacks, titles):
        fig1, plotted_image1, scale_image1 = self._create_bqplot_from_imagestack(
            imagestacks[0], titles[0]
        )
        fig2, plotted_image2, scale_image2 = self._create_bqplot_from_imagestack(
            imagestacks[1], titles[1]
        )
        figs = [fig1, fig2]
        images = [plotted_image1, plotted_image2]
        scales = [scale_image1, scale_image2]

        # slider 1 + play button
        def change_image1(change):
            plotted_image1.image = imagestacks[0][change.new]

        slider1 = IntSlider(
            value=0,
            min=0,
            max=imagestacks[0].shape[0] - 1,
            step=1,
        )
        slider1.observe(change_image1, names="value")

        play1 = Play(
            value=0,
            min=0,
            max=imagestacks[0].shape[0] - 1,
            step=1,
            interval=100,
            disabled=False,
        )
        jslink((play1, "value"), (slider1, "value"))

        # slider 2 + play button
        def change_image2(change):
            plotted_image2.image = imagestacks[1][change.new]

        slider2 = IntSlider(
            value=0,
            min=0,
            max=imagestacks[1].shape[0] - 1,
            step=1,
        )
        slider2.observe(change_image2, names="value")
        play2 = Play(
            value=0,
            min=0,
            max=imagestacks[1].shape[0] - 1,
            step=1,
            interval=100,
            disabled=False,
        )
        jslink((play2, "value"), (slider2, "value"))

        sliders = [slider1, slider2]
        plays = [play1, play2]

        return figs, images, scales, sliders, plays

    def _create_two_plots_with_single_slider(self, imagestacks, titles):

        fig1, plotted_image1, scale_image1 = self._create_bqplot_from_imagestack(
            imagestacks[0], titles[0]
        )
        fig2, plotted_image2, scale_image2 = self._create_bqplot_from_imagestack(
            imagestacks[1], titles[1]
        )
        figs = [fig1, fig2]
        images = [plotted_image1, plotted_image2]
        scales = [scale_image1, scale_image2]

        def change_image(change):
            plotted_image1.image = imagestacks[0][change.new]
            plotted_image2.image = imagestacks[1][change.new]

        slider = IntSlider(
            value=0,
            min=0,
            max=imagestacks[0].shape[0] - 1,
            step=1,
        )
        slider.observe(change_image, names="value")

        play = Play(
            value=0,
            min=0,
            max=imagestacks[0].shape[0] - 1,
            step=1,
            interval=100,
            disabled=False,
        )
        jslink((play, "value"), (slider, "value"))

        return figs, images, scales, slider, play

    def _create_bqplot_from_imagestack(self, imagestack, title="title"):
        scale_x = bq.LinearScale(min=0, max=1)
        scale_y = bq.LinearScale(min=1, max=0)
        scale_x_y = {"x": scale_x, "y": scale_y}
        fig = bq.Figure(scales=scale_x_y)
        projection_num = 0
        scale_image = {
            "x": scale_x,
            "y": scale_y,
            "image": bq.ColorScale(
                min=float(np.min(imagestack)),
                max=float(np.max(imagestack)),
                scheme="viridis",
            ),
        }
        plotted_image = ImageGL(
            image=imagestack[projection_num],
            scales=scale_image,
        )
        fig.marks = (plotted_image,)
        fig.layout.width = "550px"
        fig.layout.height = "550px"
        fig.title = f"{title}"

        return fig, plotted_image, scale_image

    def save_prj_animation(self):
        """
        Creates button to save animation. Not yet implemented.
        """

        def save_animation_on_click(click):
            os.chdir(self.Import.fpath)
            self.save_prj_animation_button.button_style = "info"
            self.save_prj_animation_button.icon = "fas fa-cog fa-spin fa-lg"
            self.save_prj_animation_button.description = "Making a movie."
            anim = self.threshold_control.save_animation(
                "projections_animation.mp4",
                self.slicer_with_hist_fig,
                "Angle",
                interval=35,
            )
            self.save_prj_animation_button.button_style = "success"
            self.save_prj_animation_button.icon = "square-check"
            self.save_prj_animation_button.description = (
                "Click again to save another animation."
            )

        self.save_prj_animation_button = Button(
            description="Click to save this animation", layout=Layout(width="auto")
        )

        self.save_prj_animation_button.on_click(save_animation_on_click)


class Center:
    """
    Class for creating a tab to help find the center of rotation. See examples
    for more information on center finding.

    Attributes
    ----------
    Import : `Import`
        Needs an import object to be constructed.
    current_center : double
        Current center of rotation. Updated when center_textbox is updated.

        TODO: this should be linked to both `Align` and `Recon`.
    center_guess : double
        Guess value for center of rotation for automatic alignment (`~tomopy.recon.rotation.find_center`).
    index_to_try : int
        Index to try out when automatically (entropy) or manually trying to
        find the center of rotation.
    search_step : double
        Step size between centers (see `tomopy.recon.rotation.write_center` or
        `tomopyui.backend.util.center`).
    search_range : double
        Will search from [center_guess - search_range] to [center_guess + search range]
        in steps of search_step.
    num_iter : int
        Number of iterations to use in center reconstruction.
    algorithm : str
        Algorithm to use in the reconstruction. Chosen from dropdown list.
    filter : str
        Filter to be used. Only works with fbp and gridrec. If you choose
        another algorith, this will be ignored.

    """

    def __init__(self, Import):

        self.Import = Import
        self.current_center = self.Import.prj_range_x[1] / 2
        self.center_guess = None
        self.index_to_try = None
        self.search_step = 0.5
        self.search_range = 5
        self.cen_range = None
        self.num_iter = 1
        self.algorithm = "gridrec"
        self.filter = "parzen"
        self.metadata = {}
        self.center_plotter = Plotter(Import=self.Import)
        self._init_widgets()
        self._set_observes()
        self.make_tab()

    def set_metadata(self):
        """
        Sets `Center` metadata.
        """

        self.metadata["center"] = self.current_center
        self.metadata["center_guess"] = self.center_guess
        self.metadata["index_to_try"] = self.index_to_try
        self.metadata["search_step"] = self.search_step
        self.metadata["search_range"] = self.search_range
        self.metadata["cen_range"] = self.cen_range
        self.metadata["num_iter"] = self.num_iter
        self.metadata["algorithm"] = self.algorithm
        self.metadata["filter"] = self.filter

    def _init_widgets(self):

        self.center_textbox = FloatText(
            description="Center: ",
            disabled=False,
            style=extend_description_style,
        )
        self.load_rough_center = Button(
            description="Click to load rough center from imported data.",
            disabled=False,
            button_style="info",
            tooltip="Loads the half-way pixel point for the center.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.center_guess_textbox = FloatText(
            description="Guess for center: ",
            disabled=False,
            style=extend_description_style,
        )
        self.find_center_button = Button(
            description="Click to automatically find center (image entropy).",
            disabled=False,
            button_style="info",
            tooltip="",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.index_to_try_textbox = IntText(
            description="Slice to use for auto:",
            disabled=False,
            style=extend_description_style,
            placeholder="Default is 1/2*y pixels",
        )
        self.num_iter_textbox = FloatText(
            description="Number of iterations: ",
            disabled=False,
            style=extend_description_style,
            value=self.num_iter,
        )
        self.search_range_textbox = IntText(
            description="Search range around center:",
            disabled=False,
            style=extend_description_style,
            value=self.search_range,
        )
        self.search_step_textbox = FloatText(
            description="Step size in search range: ",
            disabled=False,
            style=extend_description_style,
            value=self.search_step,
        )
        self.algorithms_dropdown = Dropdown(
            options=[key for key in tomopy_recon_algorithm_kwargs],
            value=self.algorithm,
            description="Algorithm:",
        )
        self.filters_dropdown = Dropdown(
            options=[key for key in tomopy_filter_names],
            value=self.filter,
            description="Algorithm:",
        )
        self.find_center_vo_button = Button(
            description="Click to automatically find center (Vo).",
            disabled=False,
            button_style="info",
            tooltip="Vo's method",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.find_center_manual_button = Button(
            description="Click to find center by plotting.",
            disabled=False,
            button_style="info",
            tooltip="Start center-finding reconstruction with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )

    def _center_update(self, change):
        self.current_center = change.new
        self.set_metadata()

    def _center_guess_update(self, change):
        self.center_guess = change.new
        self.set_metadata()

    def _load_rough_center_onclick(self, change):
        self.center_guess = self.Import.prj_range_x[1] / 2
        self.current_center = self.center_guess
        self.center_textbox.value = self.center_guess
        self.center_guess_textbox.value = self.center_guess
        self.index_to_try_textbox.value = int(np.around(self.Import.prj_range_y[1] / 2))
        self.index_to_try = self.index_to_try_textbox.value
        self.set_metadata()

    def _index_to_try_update(self, change):
        self.index_to_try = change.new
        self.set_metadata()

    def _num_iter_update(self, change):
        self.num_iter = change.new
        self.set_metadata()

    def _search_range_update(self, change):
        self.search_range = change.new
        self.set_metadata()

    def _search_step_update(self, change):
        self.search_step = change.new
        self.set_metadata()

    def _update_algorithm(self, change):
        self.algorithm = change.new
        self.set_metadata()

    def _update_filters(self, change):
        self.filter = change.new
        self.set_metadata()

    def _center_textbox_slider_update(self, change):
        self.center_textbox.value = self.cen_range[change.new]
        self.current_center = self.center_textbox.value
        self.set_metadata()

    def find_center_on_click(self, change):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center`. Takes index_to_try and center_guess.
        This method has worked better for me, if I use a good index_to_try
        and center_guess.
        """
        self.find_center_button.button_style = "info"
        self.find_center_button.icon = "fa-spin fa-cog fa-lg"
        self.find_center_button.description = "Importing data..."
        try:
            tomo = td.TomoData(metadata=self.Import.metadata)
            self.Import.log.info("Imported tomo")
            self.Import.log.info("Finding center...")
            self.Import.log.info(f"Using index: {self.index_to_try}")
        except:
            self.find_center_button.description = (
                "Please choose a file first. Try again after you do that."
            )
            self.find_center_button.button_style = "warning"
            self.find_center_button.icon = "exclamation-triangle"
        try:
            self.find_center_button.description = "Finding center..."
            self.find_center_button.button_style = "info"
            self.current_center = find_center(
                tomo.prj_imgs,
                tomo.theta,
                ratio=0.9,
                ind=self.index_to_try,
                init=self.center_guess,
            )
            self.center_textbox.value = self.current_center
            self.Import.log.info(f"Found center. {self.current_center}")
            self.find_center_button.description = "Found center."
            self.find_center_button.icon = "fa-check-square"
            self.find_center_button.button_style = "success"
        except:
            self.find_center_button.description = (
                "Something went wrong with finding center."
            )
            self.find_center_button.icon = "exclamation-triangle"
            self.find_center_button.button_style = "warning"

    def find_center_vo_on_click(self, change):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center_vo`. Note: this method has not worked
        well for me.
        """
        self.find_center_vo_button.button_style = "info"
        self.find_center_vo_button.icon = "fa-spin fa-cog fa-lg"
        self.find_center_vo_button.description = "Importing data..."
        try:
            tomo = td.TomoData(metadata=self.Import.metadata)
            self.Import.log.info("Imported tomo")
            self.Import.log.info("Finding center...")
            self.Import.log.info(f"Using index: {self.index_to_try}")
        except:
            self.find_center_vo_button.description = (
                "Please choose a file first. Try again after you do that."
            )
            self.find_center_vo_button.button_style = "warning"
            self.find_center_vo_button.icon = "exclamation-triangle"
        try:
            self.find_center_vo_button.description = "Finding center using Vo method..."
            self.find_center_vo_button.button_style = "info"
            self.current_center = find_center_vo(tomo.prj_imgs, ncore=1)
            self.center_textbox.value = self.current_center
            self.Import.log.info(f"Found center. {self.current_center}")
            self.find_center_vo_button.description = "Found center."
            self.find_center_vo_button.icon = "fa-check-square"
            self.find_center_vo_button.button_style = "success"
        except:
            self.find_center_vo_button.description = (
                "Something went wrong with finding center."
            )
            self.find_center_vo_button.icon = "exclamation-triangle"
            self.find_center_vo_button.button_style = "warning"

    def find_center_manual_on_click(self, change):
        """
        Reconstructs at various centers when you click the button, and plots
        the results with a slider so one can view. TODO: see X example.
        Uses search_range, search_step, center_guess.
        Creates a :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` +
        :doc:`histogram <mpl-interactions:examples/hist>` plot
        """
        self.find_center_manual_button.button_style = "info"
        self.find_center_manual_button.icon = "fas fa-cog fa-spin fa-lg"
        self.find_center_manual_button.description = "Starting reconstruction."

        # TODO: for memory, add only desired slice
        tomo = td.TomoData(metadata=self.Import.metadata)
        theta = tomo.theta
        cen_range = [
            self.center_guess - self.search_range,
            self.center_guess + self.search_range,
            self.search_step,
        ]

        # reconstruct, but also pull the centers used out to map to center
        # textbox
        rec, self.cen_range = write_center(
            tomo.prj_imgs,
            theta,
            cen_range=cen_range,
            ind=self.index_to_try,
            mask=True,
            algorithm=self.algorithm,
            filter_name=self.filter,
            num_iter=self.num_iter,
        )
        self.center_plotter.create_slicer_with_hist(
            plot_type="center", imagestack=rec, Center=self
        )

        # this maps the threshold_control slider to center texbox
        self.center_plotter.threshold_control.vbox.children[0].children[1].observe(
            self._center_textbox_slider_update, names="value"
        )
        self.find_center_manual_button.button_style = "success"
        self.find_center_manual_button.icon = "fa-check-square"
        self.find_center_manual_button.description = "Finished reconstruction."

        # Make VBox instantiated outside into the plot
        self.center_tab.children[2].children[0].children[2].children = [
            HBox(
                [self.center_plotter.slicer_with_hist_fig.canvas],
                layout=Layout(justify_content="center"),
            ),
            HBox(
                [
                    HBox(
                        [self.center_plotter.threshold_control_list[0]],
                        layout=Layout(align_items="center"),
                    ),
                    VBox(self.center_plotter.threshold_control_list[1::]),
                ],
                layout=Layout(justify_content="center"),
            ),
        ]
        self.manual_center_accordion = Accordion(
            children=[self.manual_center_vbox],
            selected_index=None,
            titles=("Find center through plotting",),
        )

    def _set_observes(self):
        self.center_textbox.observe(self._center_update, names="value")
        self.center_guess_textbox.observe(self._center_guess_update, names="value")
        self.load_rough_center.on_click(self._load_rough_center_onclick)
        self.index_to_try_textbox.observe(self._index_to_try_update, names="value")
        self.num_iter_textbox.observe(self._num_iter_update, names="value")
        self.search_range_textbox.observe(self._search_range_update, names="value")
        self.search_step_textbox.observe(self._search_step_update, names="value")
        self.algorithms_dropdown.observe(self._update_algorithm, names="value")
        self.filters_dropdown.observe(self._update_filters, names="value")
        self.find_center_button.on_click(self.find_center_on_click)
        self.find_center_vo_button.on_click(self.find_center_vo_on_click)
        self.find_center_manual_button.on_click(self.find_center_manual_on_click)

    def make_tab(self):
        """
        Function to create a Center object's :doc:`Tab <ipywidgets:index>`. TODO: make the tab look better.
        """

        # Accordion to find center automatically
        self.automatic_center_vbox = VBox(
            [
                HBox([self.find_center_button, self.find_center_vo_button]),
                HBox(
                    [
                        self.center_guess_textbox,
                        self.index_to_try_textbox,
                    ]
                ),
            ]
        )
        self.automatic_center_accordion = Accordion(
            children=[self.automatic_center_vbox],
            selected_index=None,
            titles=("Find center automatically",),
        )

        # Accordion to find center manually
        self.manual_center_vbox = VBox(
            [
                self.find_center_manual_button,
                HBox(
                    [
                        self.center_guess_textbox,
                        self.index_to_try_textbox,
                        self.num_iter_textbox,
                        self.search_range_textbox,
                        self.search_step_textbox,
                        self.algorithms_dropdown,
                        self.filters_dropdown,
                    ],
                    layout=Layout(
                        display="flex",
                        flex_flow="row wrap",
                        align_content="center",
                        justify_content="flex-start",
                    ),
                ),
                VBox(
                    [], layout=Layout(justify_content="center", align_content="center")
                ),
            ]
        )

        self.manual_center_accordion = Accordion(
            children=[self.manual_center_vbox],
            selected_index=None,
            titles=("Find center through plotting",),
        )

        self.center_tab = VBox(
            [
                HBox([self.center_textbox, self.load_rough_center]),
                self.automatic_center_accordion,
                self.manual_center_accordion,
            ]
        )


class Align:
    """
    Class to set automatic alignment attributes in the Alignment dashboard tab.
    On initialization, this class will create a tab.
    Features:

    - Plotting the projection images in an
      :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` +
      :doc:`histogram <mpl-interactions:examples/hist>`.
    - Selecting the plot range through the plot.
    - Selecting the plot range through range sliders.
    - Reconstruction method selection. Each :doc:`Checkbox <ipywidgets:index>` that is clicked
      will do a reconstruction using that alignment technique, and save into a
      folder with a datetime stamp.
    - Save options, including tomography images before, tomography images after
      and last reconstruction performed by the alignment.

    Attributes
    ----------
    Import : `Import`
        Needs an import object to be constructed.
    prj_shape : (Z, Y, X)
        Shape pulled from image, imported in Import tab. After choosing a file
        in Import, and choosing radio button for turning on the tab, this will load the
        projection range into the sliders.
    downsample : boolean
        Determines whether or not to downsample the data before alignment.
        Clicking this box will open another box to choose the downsample factor
        (how much you want to reduce the datasize by). This will downsample your
        data by skimage.scale TODO: check to make sure this is right.
    downsample_factor : double
        The factor used to scale the data going into the alignment routine.
        The data will be downsampled and then the alignment routine will attempt
        to correct for misalignment. After completing the alignment, the original
        (not downsampled) projection images will be shifted based on the shifts
        found in the downsampled data.
    num_iter : int
        Number of alignment iterations to perform.
    center : double
        Center of rotation used for alignment.
    upsample_factor : int
        During alignment, your data can be upsampled for sub-pixel registration
        during phase cross correlation. TODO: link to paper on this.
    extra_options : str
        This option will add the extra-options keyword argument to the tomopy
        astra wrapper. TODO: See this page: X. TODO: need a list of extra options
        keyword arguments.
    num_batches : int
        Since the data is broken up into chunks for the alignment process to
        take place on the GPU, this attribute will cut the data into chunks
        in the following way:

        .. code-block:: python

            # Jupyter
            # Cell 1
            import numpy as np
            from tomopyui.widgets.meta import Import
            a = Import()
            a
            # choose file with FileChooser in output of this cell

            # Cell 2
            a.make_tomo()
            print(a.tomo.prj_imgs.shape)
            # Output : (100, 100, 100)

            # Cell 3
            num_batches = 5
            b = np.array_split(a.tomo.prj_imgs, num_batches, axis=0)
            print(len(b))
            # Output : 5
            print(b[0].shape)
            # Output : (20, 100, 100)

    paddingX, paddingY : int
        Padding added to the projection images.
    partial : boolean
        If True, will use a partial dataset. The plot range sliders set the
        values.
    metadata : dict
        Metadata from the alignment options. This passes into
        tomopyui.backend.tomoalign.TomoAlign.

    """

    def __init__(self, Import, Center):
        self._init_attributes(Import, Center)
        self.save_opts_list = ["tomo_after", "tomo_before", "recon", "tiff", "npy"]
        self.widget_type = "Align"
        init_widgets(self)  # initialization of many widgets/some attributes
        self.set_metadata()
        self._set_metadata_obj_specific()
        self._set_observes()
        self._set_observes_obj_specific()
        self.make_tab()

    def _init_attributes(self, Import, Center):
        self.Import = Import
        self.Center = Center
        self.prj_shape = Import.prj_shape
        self.wd = Import.wd
        self.log_handler, self.log = Import.log_handler, Import.log
        self.downsample = False
        self.downsample_factor = 0.5
        self.num_iter = 1
        self.center = Center.current_center
        self.upsample_factor = 50
        self.extra_options = {}
        self.num_batches = 20
        self.prj_range_x = (0, 10)
        self.prj_range_y = (0, 10)
        self.paddingX = 10
        self.paddingY = 10
        self.partial = False
        self.tomopy_methods_list = [key for key in tomopy_recon_algorithm_kwargs]
        self.tomopy_methods_list.remove("gridrec")
        self.tomopy_methods_list.remove("fbp")
        self.astra_cuda_methods_list = [
            key for key in astra_cuda_recon_algorithm_kwargs
        ]
        self.prj_plotter = Plotter(self.Import)
        self.metadata = {}
        self.metadata["opts"] = {}
        self.run_list = []

    def set_metadata(self):
        self.metadata["opts"]["downsample"] = self.downsample
        self.metadata["opts"]["downsample_factor"] = self.downsample_factor
        self.metadata["opts"]["num_iter"] = self.num_iter
        self.metadata["opts"]["center"] = self.center
        self.metadata["opts"]["num_batches"] = self.num_batches
        self.metadata["opts"]["pad"] = (
            self.paddingX,
            self.paddingY,
        )
        self.metadata["opts"]["extra_options"] = self.extra_options
        self.metadata["methods"] = self.methods_opts
        self.metadata["save_opts"] = self.save_opts
        self.metadata["prj_range_x"] = self.prj_range_x
        self.metadata["prj_range_y"] = self.prj_range_y
        self.metadata["partial"] = self.partial

    def _set_metadata_obj_specific(self):
        self.metadata["opts"]["upsample_factor"] = self.upsample_factor

    def _set_prj_ranges_full(self):
        self.prj_range_x = (0, self.prj_shape[2] - 1)
        self.prj_range_y = (0, self.prj_shape[1] - 1)
        self.prj_range_z = (0, self.prj_shape[0] - 1)
        self.prj_range_x_slider.max = self.prj_shape[2] - 1
        self.prj_range_y_slider.max = self.prj_shape[1] - 1
        self.prj_range_x_slider.value = self.prj_range_x
        self.prj_range_y_slider.value = self.prj_range_y
        self.set_metadata()

    def load_metadata_align(self):
        self.metadata = load_metadata(self.Import.fpath_align, self.Import.fname_align)

    def _set_attributes_from_metadata(self):
        self.downsample = self.metadata["opts"]["downsample"]
        self.downsample_factor = self.metadata["opts"]["downsample_factor"]
        self.num_iter = self.metadata["opts"]["num_iter"]
        self.center = self.metadata["opts"]["center"]
        self.num_batches = self.metadata["opts"]["num_batches"]
        (self.paddingX, self.paddingY) = self.metadata["opts"]["pad"]
        self.extra_options = self.metadata["opts"]["extra_options"]
        self.methods_opts = self.metadata["methods"]
        self.save_opts = self.metadata["save_opts"]
        self.prj_range_x = self.metadata["prj_range_x"]
        self.prj_range_y = self.metadata["prj_range_y"]
        self.partial = self.metadata["partial"]

    def _set_attributes_from_metadata_obj_specific(self):
        self.upsample_factor = self.metadata["opts"]["upsample_factor"]

    # -- Radio to turn on tab ---------------------------------------------
    def _activate_tab(self, change):
        if change.new == 0:
            self.center = self.Center.current_center
            self.center_textbox.value = self.Center.current_center
            self.set_metadata()
            self.radio_fulldataset.disabled = False
            self.load_metadata_button.disabled = False
            self.start_button.disabled = False
            self.prj_shape = self.Import.prj_shape
            self.plotter_accordion.selected_index = 0
            self.save_options_accordion.selected_index = 0
            self.options_accordion.selected_index = 0
            self.methods_accordion.selected_index = 0
            self._set_prj_ranges_full()
            self.log.info("Activated alignment.")
        elif change.new == 1:
            self.radio_fulldataset.disabled = True
            self.prj_range_x_slider.disabled = True
            self.prj_range_y_slider.disabled = True
            self.load_metadata_button.disabled = True
            self.plotter_accordion.selected_index = None
            self.start_button.disabled = True
            self.save_options_accordion.selected_index = None
            self.options_accordion.selected_index = None
            self.methods_accordion.selected_index = None
            self.log.info("Deactivated alignment.")

    # -- Load metadata button ---------------------------------------------
    def _load_metadata_all_on_click(self, change):
        self.load_metadata_button.button_style = "info"
        self.load_metadata_button.icon = "fas fa-cog fa-spin fa-lg"
        self.load_metadata_button.description = "Importing metadata."
        self.load_metadata_align()
        self._set_attributes_from_metadata()
        self = _set_widgets_from_load_metadata(self)
        self._set_observes()
        self._set_observes_obj_specific()
        self.load_metadata_button.button_style = "success"
        self.load_metadata_button.icon = "fa-check-square"
        self.load_metadata_button.description = "Finished importing metadata."

    # -- Radio to turn on partial dataset ---------------------------------
    def _activate_full_partial(self, change):
        if change.new == 1:
            self.partial = True
            self.prj_range_x_slider.disabled = False
            self.prj_range_y_slider.disabled = False
            self.set_metadata()

        elif change.new == 0:
            self.partial = False
            self._set_prj_ranges_full()
            self.prj_range_x_slider.disabled = True
            self.prj_range_y_slider.disabled = True
            self.set_metadata()

    # -- Plot projections button ------------------------------------------
    def _plot_prjs_on_click(self, change):
        self.plot_prj_images_button.button_style = "info"
        self.plot_prj_images_button.icon = "fas fa-cog fa-spin fa-lg"
        self.plot_prj_images_button.description = "Importing data."
        self.prj_plotter.create_slicer_with_hist()
        self.plot_prj_images_button.button_style = "success"
        self.plot_prj_images_button.icon = "fa-check-square"
        self.plot_prj_images_button.description = "Finished Import."
        self.prj_plotter.set_range_button.description = (
            "Click to set projection range to current plot range"
        )
        self._make_prj_plot()

    def _make_prj_plot(self):
        # Make VBox inside alignment tab into the plot
        self.tab.children[1].children[0].children[1].children = [
            HBox(
                [self.prj_plotter.slicer_with_hist_fig.canvas],
                layout=Layout(justify_content="center"),
            ),
            HBox(
                [
                    HBox(
                        [self.prj_plotter.threshold_control_list[0]],
                        layout=Layout(align_items="center"),
                    ),
                    VBox(self.prj_plotter.threshold_control_list[1::]),
                ]
            ),
        ]

    # -- Button to start alignment ----------------------------------------
    def set_options_and_run(self, change):
        change.button_style = "info"
        change.icon = "fas fa-cog fa-spin fa-lg"
        change.description = (
            "Setting options and loading data into alignment algorithm."
        )
        try:
            from tomopyui.backend.tomoalign import TomoAlign

            a = TomoAlign(self)
            change.button_style = "success"
            change.icon = "fa-check-square"
            change.description = "Finished alignment."
        except:
            change.button_style = "warning"
            change.icon = "exclamation-triangle"
            change.description = "Something went wrong."

    # -- Sliders ----------------------------------------------------------
    @helpers.debounce(0.2)
    def _prj_range_x_update(self, change):
        self.prj_range_x = change.new
        self.set_metadata()

    @helpers.debounce(0.2)
    def _prj_range_y_update(self, change):
        self.prj_range_y = change.new
        self.set_metadata()

    # -- Options ----------------------------------------------------------

    # Number of iterations
    def _update_num_iter(self, change):
        self.num_iter = change.new
        self.progress_total.max = change.new
        self.set_metadata()

    # Center of rotation
    def _update_center_textbox(self, change):
        self.center = change.new
        self.set_metadata()

    # Downsampling
    def _downsample_turn_on(self, change):
        if change.new == True:
            self.downsample = True
            self.downsample_factor = self.downsample_factor_textbox.value
            self.downsample_factor_textbox.disabled = False
            self.set_metadata()
        if change.new == False:
            self.downsample = False
            self.downsample_factor = 1
            self.downsample_factor_textbox.value = 1
            self.downsample_factor_textbox.disabled = True
            self.set_metadata()

    def _update_downsample_factor_dict(self, change):
        self.downsample_factor = change.new
        self.set_metadata()

    # Upsampling
    def _update_upsample_factor(self, change):
        self.upsample_factor = change.new
        self._set_metadata_obj_specific()

    # Batch size
    def _update_num_batches(self, change):
        self.num_batches = change.new
        self.progress_phase_cross_corr.max = change.new
        self.progress_shifting.max = change.new
        self.progress_reprj.max = change.new
        self.set_metadata()

    # X Padding
    def _update_x_padding(self, change):
        self.paddingX = change.new
        self.set_metadata()

    # Y Padding
    def _update_y_padding(self, change):
        self.paddingY = change.new
        self.set_metadata()

    # Extra options
    def _update_extra_options(self, change):
        self.extra_options = change.new
        self.set_metadata()

    def _set_observes(self):

        # -- Radio to turn on tab ---------------------------------------------
        self.radio_tab.observe(self._activate_tab, names="index")

        # -- Load metadata button ---------------------------------------------
        self.load_metadata_button.on_click(self._load_metadata_all_on_click)

        # -- Plot projections button ------------------------------------------
        self.plot_prj_images_button.on_click(self._plot_prjs_on_click)

        # -- Radio to turn on partial dataset ---------------------------------
        self.radio_fulldataset.observe(self._activate_full_partial, names="index")

        # -- Sliders ----------------------------------------------------------
        self.prj_range_x_slider.observe(self._prj_range_x_update, names="value")

        self.prj_range_y_slider.observe(self._prj_range_y_update, names="value")
        # -- Options ----------------------------------------------------------

        # Center
        self.center_textbox.observe(self._update_center_textbox, names="value")

        # Downsampling
        self.downsample_checkbox.observe(self._downsample_turn_on)
        self.downsample_factor_textbox.observe(
            self._update_downsample_factor_dict, names="value"
        )

        # X Padding
        self.paddingX_textbox.observe(self._update_x_padding, names="value")

        # Y Padding
        self.paddingY_textbox.observe(self._update_y_padding, names="value")

        # Extra options
        self.extra_options_textbox.observe(self._update_extra_options, names="value")

    def _set_observes_obj_specific(self):

        # -- Set observes only for alignment ----------------------------------
        self.num_iterations_textbox.observe(self._update_num_iter, names="value")
        self.num_batches_textbox.observe(self._update_num_batches, names="value")
        self.upsample_factor_textbox.observe(
            self._update_upsample_factor, names="value"
        )
        self.start_button.on_click(self.set_options_and_run)

    def make_tab(self):
        """
        Creates an alignment tab.
        """

        # -- Saving -----------------------------------------------------------
        save_hbox = HBox(
            self.save_opts_checkboxes,
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        self.save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            titles=("Save Options",),
        )

        # -- Methods ----------------------------------------------------------
        tomopy_methods_hbox = HBox(
            [
                Label("Tomopy:", layout=Layout(width="200px", align_content="center")),
                HBox(
                    self.tomopy_methods_checkboxes,
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
            ]
        )
        astra_methods_hbox = HBox(
            [
                Label("Astra:", layout=Layout(width="100px", align_content="center")),
                HBox(
                    self.astra_cuda_methods_checkboxes,
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
            ]
        )

        recon_method_box = VBox(
            [tomopy_methods_hbox, astra_methods_hbox],
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        self.methods_accordion = Accordion(
            children=[recon_method_box], selected_index=None, titles=("Methods",)
        )

        # -- Box organization -------------------------------------------------

        pixel_range_slider_vb = VBox(
            [
                self.prj_range_x_slider,
                self.prj_range_y_slider,
            ],
            layout=Layout(width="40%"),
            justify_content="center",
            align_items="space-between",
        )

        top_of_box_hb = HBox(
            [
                self.radio_description,
                self.radio_tab,
                self.partial_radio_description,
                self.radio_fulldataset,
                pixel_range_slider_vb,
            ],
            layout=Layout(
                width="auto",
                justify_content="center",
                align_items="center",
                flex="flex-grow",
            ),
        )
        start_button_hb = HBox(
            [self.start_button], layout=Layout(width="auto", justify_content="center")
        )

        self.options_accordion = Accordion(
            children=[
                HBox(
                    [
                        self.num_iterations_textbox,
                        self.center_textbox,
                        self.upsample_factor_textbox,
                        self.num_batches_textbox,
                        self.paddingX_textbox,
                        self.paddingY_textbox,
                        self.downsample_checkbox,
                        self.downsample_factor_textbox,
                        self.extra_options_textbox,
                    ],
                    layout=Layout(
                        flex_flow="row wrap", justify_content="space-between"
                    ),
                ),
            ],
            selected_index=None,
            titles=("Alignment Options",),
        )

        progress_hbox = HBox(
            [
                self.progress_total,
                self.progress_reprj,
                self.progress_phase_cross_corr,
                self.progress_shifting,
            ]
        )

        self.tab = VBox(
            children=[
                top_of_box_hb,
                self.plotter_accordion,
                self.load_metadata_button,
                self.methods_accordion,
                self.save_options_accordion,
                self.options_accordion,
                start_button_hb,
                progress_hbox,
                VBox(
                    [self.plot_output1, self.plot_output2],
                ),
            ]
        )


class Recon(Align):
    def __init__(self, Import, Center):
        self.Center = Center
        super()._init_attributes(Import, Center)
        self.save_opts_list = ["tomo_before", "recon", "tiff", "npy"]
        self.widget_type = "Recon"
        init_widgets(self)  # initialization of many widgets/some attributes
        super().set_metadata()
        super()._set_observes()
        self._set_observes_obj_specific()
        self.make_tab()

    # -- Observe functions for reconstruction ---------------------------------

    # Start button
    def _set_options_and_run(self, change):
        change.icon = "fas fa-cog fa-spin fa-lg"
        change.description = (
            "Setting options and loading data into reconstruction algorithm(s)."
        )
        try:
            from tomopyui.backend.tomorecon import TomoRecon

            a = TomoRecon(self)
            change.button_style = "success"
            change.icon = "fa-check-square"
            change.description = "Finished reconstruction."
        except:
            change.button_style = "warning"
            change.icon = "exclamation-triangle"
            change.description = "Something went wrong."

    # Batch size
    def _update_num_batches(self, change):
        self.num_batches = change.new
        self.set_metadata()

    # Number of iterations
    def _update_num_iter(self, change):
        self.num_iter = change.new
        self.set_metadata()

    def _set_observes_obj_specific(self):

        self.start_button.on_click(self._set_options_and_run)
        self.num_batches_textbox.observe(self._update_num_batches, names="value")
        self.num_iterations_textbox.observe(self._update_num_iter, names="value")

    # -- Create recon tab -----------------------------------------------------
    def make_tab(self):

        # -- Saving -----------------------------------------------------------
        save_hbox = HBox(
            self.save_opts_checkboxes,
            layout=Layout(flex_flow="row wrap", justify_content="space-between"),
        )

        self.save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            titles=("Save Options",),
        )

        # -- Methods ----------------------------------------------------------
        tomopy_methods_hbox = HBox(
            [
                Label("Tomopy:", layout=Layout(width="200px", align_content="center")),
                HBox(
                    self.tomopy_methods_checkboxes,
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=Layout(align_content="center"),
        )
        astra_methods_hbox = HBox(
            [
                Label("Astra:", layout=Layout(width="100px", align_content="center")),
                HBox(
                    self.astra_cuda_methods_checkboxes,
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=Layout(align_content="center"),
        )

        recon_method_box = VBox(
            [tomopy_methods_hbox, astra_methods_hbox],
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        self.methods_accordion = Accordion(
            children=[recon_method_box], selected_index=None, titles=("Methods",)
        )

        # -- Box organization -------------------------------------------------

        pixel_range_slider_vb = VBox(
            [
                self.prj_range_x_slider,
                self.prj_range_y_slider,
            ],
            layout=Layout(width="30%"),
            justify_content="center",
            align_items="space-between",
        )

        top_of_box_hb = HBox(
            [
                self.radio_description,
                self.radio_tab,
                self.partial_radio_description,
                self.radio_fulldataset,
                pixel_range_slider_vb,
            ],
            layout=Layout(
                width="auto",
                justify_content="center",
                align_items="center",
                flex="flex-grow",
            ),
        )
        start_button_hb = HBox(
            [self.start_button], layout=Layout(width="auto", justify_content="center")
        )

        self.options_accordion = Accordion(
            children=[
                HBox(
                    [
                        self.num_iterations_textbox,
                        self.center_textbox,
                        self.num_batches_textbox,
                        self.paddingX_textbox,
                        self.paddingY_textbox,
                        self.downsample_checkbox,
                        self.downsample_factor_textbox,
                        self.extra_options_textbox,
                    ],
                    layout=Layout(flex_flow="row wrap"),
                ),
            ],
            selected_index=None,
            titles=("Options",),
        )

        self.tab = VBox(
            children=[
                top_of_box_hb,
                self.plotter_accordion,
                self.methods_accordion,
                self.save_options_accordion,
                self.options_accordion,
                start_button_hb,
            ]
        )


class DataExplorerTab:
    def __init__(
        self,
        Align,
        Recon,
    ):
        self.align_de = DataExplorer(Align)
        self.recon_de = DataExplorer(Recon)
        self.fb_de = DataExplorer()

    def create_data_explorer_tab(self):
        self.recent_alignment_accordion = Accordion(
            children=[self.align_de.data_plotter],
            selected_index=None,
            titles=("Plot Recent Alignments",),
        )
        self.recent_recon_accordion = Accordion(
            children=[self.recon_de.data_plotter],
            selected_index=None,
            titles=("Plot Recent Reconstructions",),
        )
        self.analysis_browser_accordion = Accordion(
            children=[self.fb_de.data_plotter],
            selected_index=None,
            titles=("Plot Any Analysis",),
        )

        self.tab = VBox(
            children=[
                self.analysis_browser_accordion,
                self.recent_alignment_accordion,
                self.recent_recon_accordion,
            ]
        )


class DataExplorer:
    def __init__(
        self, obj: (Align or Recon) = None, single_image=False, imagestacks=None
    ):
        self.figs = None
        self.single_image = single_image
        self.images = None
        self.scales = None
        self.projection_num_sliders = None
        self.imagestacks_metadata = None
        self.plays = None
        self.imagestacks = [np.zeros((15, 100, 100)) for i in range(2)]
        self.linked_stacks = False
        self.obj = obj
        self._init_widgets()

    def _init_widgets(self):
        self.button_style = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")
        self.Plotter = Plotter()
        self.app_output = Output()
        if self.obj is not None:
            if self.obj.widget_type == "Align":
                self.run_list_selector = Select(
                    options=[],
                    rows=5,
                    description="Alignments:",
                    disabled=False,
                    style=extend_description_style,
                    layout=Layout(justify_content="center"),
                )
                self.linked_stacks = True
                self.titles = ["Before Alignment", "After Alignment"]
                self.load_run_list_button = Button(
                    description="Load alignment list",
                    icon="download",
                    button_style="info",
                    layout=Layout(width="auto"),
                )
            else:
                self.run_list_selector = Select(
                    options=[],
                    rows=5,
                    description="Reconstructions:",
                    disabled=False,
                    style=extend_description_style,
                    layout=Layout(justify_content="center"),
                )
                self.linked_stacks = False
                self.titles = ["Projections", "Reconstruction"]
                self.load_run_list_button = Button(
                    description="Load reconstruction list",
                    icon="download",
                    button_style="info",
                    layout=Layout(width="auto"),
                )

            self.run_list_selector.observe(self.choose_file_to_plot, names="value")
            self.load_run_list_button.on_click(self._load_run_list_on_click)
            self._create_plotter_run_list()

        elif self.single_image:
            self.titles = ["Projections"]
        else:

            self.titles = ["Projections", "Reconstruction"]
            self.filebrowser = Filebrowser()
            self.filebrowser.create_file_browser()
            self.filebrowser.load_data_button.on_click(self.load_data_from_filebrowser)
            self._create_plotter_filebrowser()

    def load_data_from_filebrowser(self, change):
        metadata = {}
        metadata["fpath"] = self.filebrowser.metadata["parent_fpath"]
        metadata["fname"] = self.filebrowser.metadata["parent_fname"]
        metadata["angle_start"] = self.filebrowser.metadata["angle_start"]
        metadata["angle_end"] = self.filebrowser.metadata["angle_end"]
        tomo = TomoData(metadata=metadata)
        self.imagestacks[0] = tomo.prj_imgs
        metadata["fpath"] = str(self.filebrowser.selected_method)
        metadata["fname"] = str(self.filebrowser.selected_data_fname)
        tomo = TomoData(metadata=metadata)
        # TODO: make this agnostic to recon/tomo
        self.imagestacks[1] = tomo.prj_imgs
        if self.filebrowser.selected_analysis_type == "recon":
            self.titles = ["Projections", "Reconstruction"]
        else:
            self.titles = ["Before Alignment", "After Alignment"]
        self.create_figures_and_widgets()
        self._create_image_app()

    def find_file_in_metadata(self, foldername):
        for run in range(len(self.obj.run_list)):
            if foldername in self.obj.run_list[run]:
                metadata = {}
                metadata["fpath"] = self.obj.run_list[run][foldername]["parent_fpath"]
                metadata["fname"] = self.obj.run_list[run][foldername]["parent_fname"]
                metadata["angle_start"] = self.obj.run_list[run][foldername][
                    "angle_start"
                ]
                metadata["angle_end"] = self.obj.run_list[run][foldername]["angle_end"]
                self.imagestacks[0] = TomoData(metadata=metadata).prj_imgs
                metadata["fpath"] = self.obj.run_list[run][foldername]["savedir"]
                if self.obj.widget_type == "Align":
                    metadata["fname"] = "projections_after_alignment.tif"
                else:
                    metadata["fname"] = "recon.tif"
                self.imagestacks[1] = TomoData(metadata=metadata).prj_imgs
                self.create_figures_and_widgets()
                self._create_image_app()

    def create_figures_and_widgets(self):
        if self.single_image:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_one_plot(self.imagestacks, self.titles)
        elif self.linked_stacks:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_two_plots_with_single_slider(
                self.imagestacks, self.titles
            )
            self.projection_num_sliders = [self.projection_num_sliders]
            self.plays = [self.plays]

        else:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_two_plots_with_two_sliders(
                self.imagestacks, self.titles
            )

        self.vmin_vmax_sliders = self._create_vmin_vmax_sliders()
        self.remove_high_low_intensity_buttons = (
            self._create_remove_high_low_intensity_buttons()
        )
        self.swapaxes_buttons = self._create_swapaxes_buttons()
        self.reset_button = Button(
            icon="redo", style=self.button_style, layout=self.button_layout
        )
        self.reset_button.on_click(self._reset_on_click)

        self.reset_button_one_fig = Button(
            icon="redo", style=self.button_style, layout=self.button_layout
        )
        self.reset_button_one_fig.on_click(self._reset_on_click_one_fig)

    def _create_vmin_vmax_sliders(self):
        vmin = self.imagestacks[0].min()
        vmax = self.imagestacks[0].max()
        slider1 = FloatRangeSlider(
            description="vmin-vmax:",
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 1000,
            value=(vmin, vmax),
            orientation="vertical",
        )

        def change_vmin_vmax1(change):
            self.scales[0]["image"].min = change["new"][0]
            self.scales[0]["image"].max = change["new"][1]

        slider1.observe(change_vmin_vmax1, names="value")

        vmin = self.imagestacks[1].min()
        vmax = self.imagestacks[1].max()
        slider2 = FloatRangeSlider(
            description="vmin-vmax:",
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 1000,
            value=(vmin, vmax),
            orientation="vertical",
        )

        def change_vmin_vmax2(change):
            self.scales[1]["image"].min = change["new"][0]
            self.scales[1]["image"].max = change["new"][1]

        slider2.observe(change_vmin_vmax2, names="value")

        sliders = [slider1, slider2]

        return sliders

    def _create_swapaxes_buttons(self):
        def swapaxes_on_click1(change):
            # defaults to going with the high/low value from
            self.imagestacks[0] = self.Plotter._swap_axes_on_click(
                self.imagestacks[0],
                self.images[0],
                self.projection_num_sliders[0],
            )

        def swapaxes_on_click2(change):
            # defaults to going with the high/low value from
            self.imagestacks[1] = self.Plotter._swap_axes_on_click(
                self.imagestacks[1],
                self.images[1],
                self.projection_num_sliders[1],
            )

        button1 = Button(
            icon="random", layout=self.button_layout, style=self.button_style
        )
        # button1.button_style = "info"
        button2 = Button(
            icon="random", layout=self.button_layout, style=self.button_style
        )
        # button2.button_style = "info"
        button1.on_click(swapaxes_on_click1)
        button2.on_click(swapaxes_on_click2)
        buttons = [button1, button2]
        return buttons

    def _create_remove_high_low_intensity_buttons(self):
        """
        Parameters
        ----------
        imagestack: np.ndarray
            images that it will use to find vmin, vmax - this is found by
            getting the 0.5 and 99.5 percentiles of the data
        scale: dict

        """

        def remove_high_low_intensity_on_click1(change):
            # defaults to going with the high/low value from
            self.Plotter._remove_high_low_intensity_on_click(
                self.imagestacks[0], self.scales[0], self.vmin_vmax_sliders[0]
            )

        def remove_high_low_intensity_on_click2(change):
            # defaults to going with the high/low value from
            self.Plotter._remove_high_low_intensity_on_click(
                self.imagestacks[1], self.scales[1], self.vmin_vmax_sliders[1]
            )

        button1 = Button(
            icon="adjust", layout=self.button_layout, style=self.button_style
        )
        button1.button_style = "info"
        button2 = Button(
            icon="adjust", layout=self.button_layout, style=self.button_style
        )
        button2.button_style = "info"
        button1.on_click(remove_high_low_intensity_on_click1)
        button2.on_click(remove_high_low_intensity_on_click2)
        buttons = [button1, button2]
        return buttons

    def choose_file_to_plot(self, change):
        self.find_file_in_metadata(change.new)

    def _load_run_list_on_click(self, change):
        self.load_run_list_button.button_style = "info"
        self.load_run_list_button.icon = "fas fa-cog fa-spin fa-lg"
        self.load_run_list_button.description = "Importing run list."
        # creates a list from the keys in pythonic way
        # from https://stackoverflow.com/questions/11399384/extract-all-keys-from-a-list-of-dictionaries
        # don't know how it works
        self.run_list_selector.options = list(
            set().union(*(d.keys() for d in self.obj.run_list))
        )
        self.load_run_list_button.button_style = "success"
        self.load_run_list_button.icon = "fa-check-square"
        self.load_run_list_button.description = "Finished importing run list."

    def _reset_on_click(self, change):
        self.create_figures_and_widgets()
        self._create_image_app()

    def _reset_on_click_one_fig(self, change):
        self.create_figures_and_widgets()
        self._create_image_app_raw_import()

    def _create_image_app(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        footer_layout = Layout(justify_content="center")
        header = None

        self.button_box1 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[0],
                self.swapaxes_buttons[0],
            ],
            layout=left_sidebar_layout,
        )
        self.button_box2 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[1],
                self.swapaxes_buttons[1],
            ],
            layout=right_sidebar_layout,
        )

        left_sidebar = VBox(
            [self.vmin_vmax_sliders[0], self.button_box1], layout=left_sidebar_layout
        )
        center = HBox(self.figs, layout=Layout(justify_content="center"))
        right_sidebar = VBox(
            [self.vmin_vmax_sliders[1], self.button_box2], layout=right_sidebar_layout
        )
        if self.linked_stacks:
            footer = HBox(
                self.plays + self.projection_num_sliders, layout=footer_layout
            )
        else:
            footer = HBox(
                [
                    HBox([self.plays[0], self.projection_num_sliders[0]]),
                    HBox([self.plays[1], self.projection_num_sliders[1]]),
                ],
                layout=footer_layout,
            )
        self.image_app = AppLayout(
            header=header,
            left_sidebar=left_sidebar,
            center=center,
            right_sidebar=right_sidebar,
            footer=footer,
            pane_widths=[0.5, 5, 0.5],
            pane_heights=[0, 10, "40px"],
            height="auto",
        )
        with self.app_output:
            self.app_output.clear_output(wait=True)
            display(self.image_app)

    def _create_image_app_raw_import(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        footer_layout = Layout(justify_content="center")
        header = None

        self.button_box1 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[0],
                self.swapaxes_buttons[0],
            ],
            layout=left_sidebar_layout,
        )

        left_sidebar = VBox(
            [self.vmin_vmax_sliders[0], self.button_box1], layout=left_sidebar_layout
        )
        center = HBox(self.figs, layout=Layout(justify_content="center"))
        right_sidebar = None

        if self.linked_stacks:
            footer = HBox(
                self.plays + self.projection_num_sliders, layout=footer_layout
            )

        self.image_app = AppLayout(
            header=header,
            left_sidebar=left_sidebar,
            center=center,
            right_sidebar=right_sidebar,
            footer=footer,
            pane_widths=[0.5, 5, 0.5],
            pane_heights=[0, 10, "40px"],
            height="auto",
        )
        with self.app_output:
            self.app_output.clear_output(wait=True)
            display(self.image_app)

    def _create_plotter_run_list(self):
        # self.create_figures_and_widgets()
        # self._create_image_app()
        self.data_plotter = VBox(
            [self.load_run_list_button, self.run_list_selector, self.app_output]
        )

    def _create_plotter_filebrowser(self):
        # self.create_figures_and_widgets()
        # self._create_image_app()
        self.data_plotter = VBox([self.filebrowser.filebrowser, self.app_output])


class Filebrowser:
    def __init__(self):

        # parent directory filechooser
        self.orig_data_fc = FileChooser()
        self.orig_data_fc.register_callback(self.update_orig_data_folder)
        self.fc_label = Label("Original Data", layout=Layout(justify_content="Center"))

        # subdirectory selector
        self.subdir_list = []
        self.subdir_label = Label(
            "Analysis Directories", layout=Layout(justify_content="Center")
        )
        self.subdir_selector = Select(options=self.subdir_list, rows=5, disabled=False)
        self.subdir_selector.observe(self.populate_methods_list, names="value")
        self.selected_subdir = None

        # method selector
        self.methods_label = Label("Methods", layout=Layout(justify_content="Center"))
        self.methods_list = []
        self.methods_selector = Select(
            options=self.methods_list, rows=5, disabled=False
        )
        self.methods_selector.observe(self.populate_data_list, names="value")
        self.selected_method = None

        # data selector
        self.data_label = Label("Data", layout=Layout(justify_content="Center"))
        self.data_list = []
        self.data_selector = Select(options=self.data_list, rows=5, disabled=False)
        self.data_selector.observe(self.set_data_filename, names="value")
        self.allowed_extensions = (".npy", ".tif", ".tiff")
        self.selected_data_fname = None
        self.selected_data_ftype = None
        self.selected_analysis_type = None

        self.options_metadata_table_output = Output()

        # load data button
        self.load_data_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="info",
            layout=Layout(width="75px", height="86px"),
        )

    def populate_subdirs_list(self):
        self.subdir_list = [
            pathlib.PurePath(f) for f in os.scandir(self.root_fpath) if f.is_dir()
        ]
        self.subdir_list = [
            subdir.parts[-1]
            for subdir in self.subdir_list
            if any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.subdir_selector.options = self.subdir_list

    def update_orig_data_folder(self):
        self.root_fpath = self.orig_data_fc.selected_path
        self.populate_subdirs_list()
        self.methods_selector.options = []

    def populate_methods_list(self, change):
        self.selected_subdir = pathlib.PurePath(self.root_fpath) / change.new
        self.methods_list = [
            pathlib.PurePath(f) for f in os.scandir(self.selected_subdir) if f.is_dir()
        ]
        self.methods_list = [
            subdir.parts[-1]
            for subdir in self.methods_list
            if not any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.methods_selector.options = self.methods_list

    def populate_data_list(self, change):
        if change.new is not None:
            self.selected_method = (
                pathlib.PurePath(self.root_fpath) / self.selected_subdir / change.new
            )
            self.file_list = [
                pathlib.PurePath(f)
                for f in os.scandir(self.selected_method)
                if not f.is_dir()
            ]
            self.data_list = [
                file.name
                for file in self.file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            self.data_selector.options = self.data_list
            self.load_metadata()
        else:
            self.data_selector.options = []

    def set_data_filename(self, change):
        self.selected_data_fname = change.new
        self.selected_data_ftype = pathlib.PurePath(self.selected_data_fname).suffix
        if "recon" in pathlib.PurePath(self.selected_subdir).name:
            self.selected_analysis_type = "recon"
        elif "align" in pathlib.PurePath(self.selected_subdir).name:
            self.selected_analysis_type = "align"

    def load_metadata(self):
        self.metadata_file = [
            self.selected_method / file.name
            for file in self.file_list
            if "metadata.json" in file.name
        ]
        if self.metadata_file != []:
            self.metadata = load_metadata(fullpath=self.metadata_file[0])
            self.options_table = metadata_to_DataFrame(self.metadata)
            with self.options_metadata_table_output:
                self.options_metadata_table_output.clear_output(wait=True)
                display(self.options_table)

    def create_file_browser(self):
        fc = VBox([self.fc_label, self.orig_data_fc])
        subdir = VBox([self.subdir_label, self.subdir_selector])
        methods = VBox([self.methods_label, self.methods_selector])
        data = VBox([self.data_label, self.data_selector])
        button = VBox(
            [
                Label("Upload", layout=Layout(justify_content="center")),
                self.load_data_button,
            ]
        )
        top_hb = HBox(
            [fc, subdir, methods, data, button],
            layout=Layout(justify_content="center"),
            align_items="stretch",
        )
        box = VBox(
            [top_hb, self.options_metadata_table_output],
            layout=Layout(justify_content="center", align_items="center"),
        )
        self.filebrowser = box
