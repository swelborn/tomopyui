#!/usr/bin/env python

from ipywidgets import *
from ._import import import_helpers
from ._shared import helpers
from ._shared._init_widgets import init_widgets
from ipyfilechooser import FileChooser
from mpl_interactions import hyperslicer, ioff, interactive_hist, zoom_factory
from tomopyui.backend.util.center import write_center
from tomopy.recon.rotation import find_center_vo, find_center, find_center_pc

import functools
import os
import tomopy.prep.normalize
import tomopyui.backend.tomodata as td
import matplotlib.pyplot as plt
import numpy as np
import logging

extend_description_style = {"description_width": "auto"}


class Import:

    def __init__(self):

        # Init textboxes
        self.angle_start = -90
        self.angle_end = 90
        self.num_theta = 360
        self.image_shape = None
        self.angles_textboxes = import_helpers.create_angles_textboxes(self)

        # Init checkboxes
        self.import_opts_list = ["rotate"]
        self.import_opts = {key: False for key in self.import_opts_list}
        self.opts_checkboxes = helpers.create_checkboxes_from_opt_list(
                                                    self.import_opts_list, 
                                                    self.import_opts,
                                                    self)

        # Init filechooser
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_file_information)
        self.prj_range_x = None
        self.prj_range_y = None
        self.prj_range_z = None
        self.tomo = None
        self.wd = None

        # Init logger to be used throughout the app. 
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(
            self.log, logging_level=20
        )

        # Init metadata
        self.metadata = {}
        self.set_metadata()  # init metadata

    def set_wd(self, wd):
        self.wd = wd

    def set_metadata(self):
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta,
            "prj_range_x" : self.prj_range_x,
            "prj_range_y" : self.prj_range_y,
        } | self.import_opts

    # TODO: 
    def load_metadata(self):
        print("load metadata here")

    def update_file_information(self):
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_wd(self.fpath)
        # metadata must be set here in case tomodata is created (for folder 
        # import). this can be changed later.
        self.set_metadata()
        self.get_image_shape()
        self.set_prj_ranges()
        self.set_metadata()

    def get_image_shape(self):
        if self.fname.__contains__(".tif"):
            self.image_shape = helpers.get_img_shape(self.fpath, 
                                                    self.fname,
                                                    "tiff",
                                                    self.metadata,
                                                    )
        elif self.fname.__contains__(".npy"):
            self.image_shape = helpers.get_img_shape(self.fpath, 
                                                    self.fname,
                                                    "npy",
                                                    self.metadata,
                                                    )
        elif self.fname == "":
            self.image_shape = helpers.get_img_shape(self.fpath, 
                                                    self.fname,
                                                    "tiff",
                                                    self.metadata,
                                                    folder_import=True
                                                    )

    def set_prj_ranges(self):
        self.prj_range_x = (0, self.image_shape[2]-1)
        self.prj_range_y = (0, self.image_shape[1]-1)
        self.prj_range_z = (0, self.image_shape[0]-1)

    def make_tomo(self):
        self.tomo = td.TomoData(metadata=self.metadata)

class Plotter:
    def __init__(self, Import):

        self.Import = Import
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
            layout=Layout(width="100%"),
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
            layout=Layout(width="100%"),
            style=extend_description_style,
        )
        self.slicer_with_hist_fig = None
        self.threshold_control = None
        self.threshold_control_list = None
        self.set_range_button = Button(
            description="Click to set current range to plot range.",
            layout=Layout(width="auto"),
        )
        self.link_ranges_button_alignment = None
        self.link_ranges_button_recon = None
        self.save_animation_button = None

    def create_slicer_with_hist(
        self, plot_type="prj", imagestack=None, Center=None
    ):
        """
        Creates a plot with a histogram for a given set of data

        Parameters
        -----------
        plot_type:str. Used for deciding what to use for the titles of things, etc.
        """

        # Turn off immediate display of plot.
        with ioff:
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

        # allowing zoom on scroll. TODO: figure out panning without clicking
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

    def save_prj_animation(self):
        """
        Creates button to save animation.
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


class Prep:
    def __init__(self, Import):

        self.tomo = Import.tomo
        self.dark = None
        self.flat = None
        self.darkfc = FileChooser()
        self.darkfc.register_callback(self.set_fpath_dark)
        self.flatfc = FileChooser()
        self.flatfc.register_callback(self.set_fpath_flat)
        self.fpathdark = None
        self.fnamedark = None
        self.fpathflat = None
        self.fnameflat = None
        self.rotate = Import.metadata["rotate"]
        self.set_metadata_dark()
        self.set_metadata_flat()

    def set_metadata_dark(self):
        self.darkmetadata = {
            "fpath": self.fpathdark,
            "fname": self.fnamedark,
            "opts": {"rotate": self.rotate},
        }

    def set_metadata_flat(self):
        self.flatmetadata = {
            "fpath": self.fpathflat,
            "fname": self.fnameflat,
            "opts": {"rotate": self.rotate},
        }

    def set_fpath_dark(self):
        self.fpathdark = self.darkfc.selected_path
        self.fnamedark = self.darkfc.selected_filename
        self.set_metadata()

    def set_fpath_flat(self):
        self.fpathflat = self.flatfc.selected_path
        self.fnameflat = self.flatfc.selected_filename
        self.set_metadata()

    def normalize(self, rm_zeros_nans=True):
        tomo_norm = tomopy.prep.normalize.normalize(
            self.tomo.prj_imgs, self.flat.prj_imgs, self.dark.prj_imgs
        )
        tomo_norm = td.TomoData(prj_imgs=prj_imgs, raw="No")
        tomo_norm_mlog = tomopy.prep.normalize.minus_log(tomo_norm)
        tomo_norm_mlog = td.TomoData(prj_imgs=tomoNormMLogprj_imgs, raw="No")
        if rm_zeros_nans == True:
            tomo_norm_mlog.prj_imgs = tomopy.misc.corr.remove_nan(
                tomo_norm_mlog.prj_imgs, val=0.0
            )
            tomo_norm_mlog.prj_imgs[tomo_norm_mlog.prj_imgs == np.inf] = 0
        self.tomo = tomo_norm_mlog

class Center:
    def __init__(self, Import, Prep=None, Align=None, Recon=None):
        self.Import = Import
        self.Prep = Prep
        self.Align = Align
        self.Recon = Recon
        # add a method in Import prj range (just like in Align/Prep)
        self.current_center = 5
        self.center_guess = self.current_center
        self.centers_list = None
        self.index_to_try = 50
        self.search_step = 0.5
        self.search_range = 5
        self.center_textbox = None
        self.find_center_button = None
        self.find_center_vo_button = None
        self.cen_range = None
        self.num_iter = 1
        self.algorithm = "gridrec"
        self.filter = "parzen"
        self.err_output = Output()
        self.manual_center_vbox = VBox([])
        self.automatic_center_vbox = VBox([])
        self.automatic_center_accordion = Accordion(children=[])
        self.manual_center_accordion = Accordion(children=[])
        self.center_tab = Tab(children=[])
        self.center_plotter = Plotter(Import=self.Import)
        self.create_center_tab()

    def create_center_tab(self):
        def find_center_on_click(click):
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

        self.find_center_button = Button(
            description="Click to automatically find center (image entropy).",
            disabled=False,
            button_style="info",
            tooltip="",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.find_center_button.on_click(find_center_on_click)

        def find_center_vo_on_click(click):
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
                self.find_center_vo_button.description = (
                    "Finding center using Vo method..."
                )
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

        self.find_center_vo_button = Button(
            description="Click to automatically find center (Vo).",
            disabled=False,
            button_style="info",
            tooltip="Vo's method",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.find_center_vo_button.on_click(find_center_vo_on_click)

        def center_textbox_slider_update(change):
            self.center_textbox.value = self.cen_range[change.new]

        def find_center_manual_on_click(change):
            self.find_center_manual_button.button_style = "info"
            self.find_center_manual_button.icon = "fas fa-cog fa-spin fa-lg"
            self.find_center_manual_button.description = "Starting reconstruction."

            # TODO: for memory, add only desired slice
            tomo = td.TomoData(metadata=self.Import.metadata)
            theta = tomo.theta
            cen_range = [
                self.current_center - self.search_range,
                self.current_center + self.search_range,
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
                center_textbox_slider_update, names="value"
            )
            self.find_center_manual_button.button_style = "success"
            self.find_center_manual_button.icon = "fa-check-square"
            self.find_center_manual_button.description = "Finished reconstruction."

            # Make VBox instantiated outside into the plot
            self.center_tab.children[2].children[0].children[2].children = [
                HBox([self.center_plotter.slicer_with_hist_fig.canvas]),
                HBox(self.center_plotter.threshold_control_list),
            ]
            self.manual_center_accordion = Accordion(
                children=[self.manual_center_vbox],
                selected_index=None,
                titles=("Find center through plotting",),
            )

        self.find_center_manual_button = Button(
            description="Click to find center by plotting.",
            disabled=False,
            button_style="info",
            tooltip="Start center-finding reconstruction with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.find_center_manual_button.on_click(find_center_manual_on_click)

        def center_update(change):
            self.current_center = change.new
            # self.Import.center = change.new
            # self.Prep.center = change.new
            # self.Recon.center = change.new

        self.center_textbox = FloatText(
            description="Center: ",
            disabled=False,
            style=extend_description_style,
            value=self.current_center,
        )
        self.center_textbox.observe(center_update, names="value")

        def center_guess_update(change):
            self.center_guess = change.new

        center_guess_textbox = FloatText(
            description="Guess for center: ",
            disabled=False,
            style=extend_description_style,
        )
        center_guess_textbox.observe(center_guess_update, names="value")

        def search_step_update(change):
            self.search_step = change.new

        search_step_textbox = FloatText(
            description="Step size in search range: ",
            disabled=False,
            style=extend_description_style,
            value=self.search_step,
        )
        search_step_textbox.observe(search_step_update, names="value")

        def search_range_update(change):
            self.search_range = change.new

        search_range_textbox = IntText(
            description="Search range around center:",
            disabled=False,
            style=extend_description_style,
            value=self.search_range,
        )
        search_range_textbox.observe(search_range_update, names="value")

        allowed_recon_kwargs = {
            "art": ["num_gridx", "num_gridy", "num_iter"],
            "bart": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
            "fbp": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
            "gridrec": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
            "mlem": ["num_gridx", "num_gridy", "num_iter"],
            "osem": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
            "ospml_hybrid": [
                "num_gridx",
                "num_gridy",
                "num_iter",
                "reg_par",
                "num_block",
                "ind_block",
            ],
            "ospml_quad": [
                "num_gridx",
                "num_gridy",
                "num_iter",
                "reg_par",
                "num_block",
                "ind_block",
            ],
            "pml_hybrid": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
            "pml_quad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
            "sirt": ["num_gridx", "num_gridy", "num_iter"],
            "tv": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
            "grad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
            "tikh": ["num_gridx", "num_gridy", "num_iter", "reg_data", "reg_par"],
        }

        filter_names = {
            "none",
            "shepp",
            "cosine",
            "hann",
            "hamming",
            "ramlak",
            "parzen",
            "butterworth",
        }

        def update_filters(change):
            self.filter = change.new

        filters_dropdown = Dropdown(
            options=[key for key in filter_names],
            value=self.filter,
            description="Algorithm:",
        )
        filters_dropdown.observe(update_filters, names="value")

        def update_algorithm(change):
            self.algorithm = change.new

        algorithms_dropdown = Dropdown(
            options=[key for key in allowed_recon_kwargs],
            value=self.algorithm,
            description="Algorithm:",
        )
        algorithms_dropdown.observe(update_algorithm, names="value")

        def num_iter_update(change):
            self.num_iter = change.new

        num_iter_textbox = FloatText(
            description="Number of iterations: ",
            disabled=False,
            style=extend_description_style,
            value=self.num_iter,
        )
        num_iter_textbox.observe(num_iter_update, names="value")

        def index_to_try_update(change):
            self.index_to_try = change.new

        index_to_try_textbox = IntText(
            description="Projection image to use for auto:",
            disabled=False,
            style=extend_description_style,
            value=self.index_to_try,
        )
        index_to_try_textbox.observe(index_to_try_update, names="value")

        # Accordion to find center automatically
        self.automatic_center_vbox = VBox(
            [
                HBox([self.find_center_button, self.find_center_vo_button]),
                HBox(
                    [
                        center_guess_textbox,
                        index_to_try_textbox,
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
                        index_to_try_textbox,
                        num_iter_textbox,
                        search_range_textbox,
                        search_step_textbox,
                        algorithms_dropdown,
                        filters_dropdown,
                    ]
                ),
                VBox([]),
            ]
        )

        self.manual_center_accordion = Accordion(
            children=[self.manual_center_vbox],
            selected_index=None,
            titles=("Find center through plotting",),
        )

        self.center_tab = VBox(
            [
                self.center_textbox,
                self.automatic_center_accordion,
                self.manual_center_accordion,
                self.err_output,
            ]
        )

class Align:

    def __init__(self, Import):

        self.init_attributes(Import)
        self.save_opts_list = ["tomo_after", "tomo_before", "recon", "tiff", 
                                "npy"]
        self.widget_type = "Align"
        init_widgets(self) # initialization of many widgets/some attributes
        self.set_metadata()
        self.set_metadata_obj_specific()
        self.set_observes()
        self.set_observes_obj_specific()
        self.make_tab()

    def init_attributes(self, Import):
        self.Import = Import
        self.image_shape = Import.image_shape
        self.angle_start = Import.angle_start # TODO
        self.angle_end = Import.angle_end # TODO
        self.num_theta = Import.num_theta # TODO
        self.wd = Import.wd
        self.log_handler, self.log = Import.log_handler, Import.log
        self.downsample = False
        self.downsample_factor = 0.5
        self.num_iter = 1
        self.center = 0
        self.upsample_factor = 50
        self.extra_options = {}
        self.batch_size = 20
        self.prj_range_x = (0, 10)
        self.prj_range_y = (0, 10)
        self.paddingX = 10
        self.paddingY = 10
        self.partial = False
        self.methods_list = ["FBP_CUDA", "SIRT_CUDA", "SART_CUDA", "CGLS_CUDA",
                            "MLEM_CUDA","SIRT Plugin-Faster","SIRT 3D-Fastest"]
        self.prj_plotter = Plotter(self.Import)
        self.metadata = {}
        self.metadata["opts"] = {}

    def set_metadata(self):

        self.metadata["opts"]["downsample"] = self.downsample
        self.metadata["opts"]["downsample_factor"] = self.downsample_factor
        self.metadata["opts"]["num_iter"] = self.num_iter
        self.metadata["opts"]["center"] = self.center
        self.metadata["opts"]["batch_size"] = self.batch_size
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

    def set_metadata_obj_specific(self):
        self.metadata["opts"]["upsample_factor"] = self.upsample_factor

    def set_prj_ranges_full(self):

        self.prj_range_x = (0, self.image_shape[2]-1)
        self.prj_range_y = (0, self.image_shape[1]-1)
        self.prj_range_z = (0, self.image_shape[0]-1)
        self.prj_range_x_slider.max = self.image_shape[2]-1
        self.prj_range_y_slider.max = self.image_shape[1]-1
        self.prj_range_x_slider.value = self.prj_range_x
        self.prj_range_y_slider.value = self.prj_range_y
        self.set_metadata()

    # -- Radio to turn on tab ---------------------------------------------
    def activate_tab(self, change):
        if change.new == 0:
            self.radio_fulldataset.disabled = False
            self.image_shape = self.Import.image_shape
            self.plotter_accordion.selected_index = 0
            self.start_button.disabled = False
            self.save_options_accordion.selected_index = 0
            self.options_accordion.selected_index = 0
            self.methods_accordion.selected_index = 0
            self.set_prj_ranges_full()
            self.log.info("Activated alignment.")
        elif change.new == 1:
            self.radio_fulldataset.disabled = True
            self.prj_range_x_slider.disabled = True
            self.prj_range_y_slider.disabled = True
            self.plotter_accordion.selected_index = None
            self.start_button.disabled = True
            self.save_options_accordion.selected_index = None
            self.options_accordion.selected_index = None
            self.methods_accordion.selected_index = None
            self.log.info("Deactivated alignment.")

    # -- Radio to turn on partial dataset ---------------------------------
    def activate_full_partial(self, change):
        if change.new == 1:
            self.partial = True
            self.prj_range_x_slider.disabled = False
            self.prj_range_y_slider.disabled = False
            self.set_metadata()

        elif change.new == 0:
            self.partial = False
            self.set_prj_ranges_full()
            self.prj_range_x_slider.disabled = True
            self.prj_range_y_slider.disabled = True
            self.set_metadata()

    # -- Plot projections button ------------------------------------------
    def plot_prjs_on_click(self, change):
        self.plot_prj_images_button.button_style = "info"
        self.plot_prj_images_button.icon = "fas fa-cog fa-spin fa-lg"
        self.plot_prj_images_button.description = "Importing data."
        self.prj_plotter.create_slicer_with_hist()
        self.plot_prj_images_button.button_style = "success"
        self.plot_prj_images_button.icon = "fa-check-square"
        self.plot_prj_images_button.description = "Finished Import."
        self.prj_plotter.set_range_button.description = "Click to set projection range to current plot range"
        self.make_prj_plot()

    def make_prj_plot(self):
        # Make VBox inside alignment tab into the plot
        self.tab.children[1].children[0].children[1].children = [
            HBox([self.prj_plotter.slicer_with_hist_fig.canvas]),
            HBox(self.prj_plotter.threshold_control_list),
        ]

    # -- Button to start alignment ----------------------------------------
    def set_options_and_run(self, change):
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
    def prj_range_x_update(self, change):
        self.prj_range_x = change.new
        self.set_metadata()

    @helpers.debounce(0.2)
    def prj_range_y_update(self, change):
        self.prj_range_y = change.new
        self.set_metadata()
    
    # -- Options ---------------------------------------------------------- 

    # Number of iterations
    def update_num_iter(self, change):
        self.num_iter = change.new
        self.progress_total.max = change.new
        self.set_metadata()

    # Center of rotation
    def update_center_of_rotation(self, change):
        self.center = change.new
        self.set_metadata()

    # Downsampling
    def downsample_turn_on(self, change):
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

    def update_downsample_factor_dict(self, change):
        self.downsample_factor = change.new
        self.set_metadata()

    # Upsampling
    def update_upsample_factor(self, change):
        self.upsample_factor = change.new
        self.set_metadata_obj_specific()

    # Batch size
    def update_batch_size(self, change):
        self.batch_size = change.new
        self.progress_phase_cross_corr.max = change.new
        self.progress_shifting.max = change.new
        self.progress_reprj.max = change.new
        self.set_metadata()

    # X Padding
    def update_x_padding(self, change):
        self.paddingX = change.new
        self.set_metadata()

    # Y Padding
    def update_y_padding(self, change):
        self.paddingY = change.new
        self.set_metadata()

    # Extra options
    def update_extra_options(self, change):
        self.extra_options = change.new
        self.set_metadata()

    def set_observes(self):

        # -- Radio to turn on tab ---------------------------------------------
        self.radio_tab.observe(self.activate_tab, names="index")

        # -- Plot projections button ------------------------------------------
        self.plot_prj_images_button.on_click(self.plot_prjs_on_click)

        # -- Radio to turn on partial dataset ---------------------------------
        self.radio_fulldataset.observe(self.activate_full_partial, names="index")

        # -- Sliders ----------------------------------------------------------
        self.prj_range_x_slider.observe(
            self.prj_range_x_update, names="value"
        )

        self.prj_range_y_slider.observe(
            self.prj_range_y_update, names="value"
        )
        # -- Options ----------------------------------------------------------

        # Center 
        self.center_of_rotation.observe(self.update_center_of_rotation, names="value")

        # Downsampling
        self.downsample_checkbox.observe(self.downsample_turn_on)
        self.downsample_factor_textbox.observe(self.update_downsample_factor_dict, names="value")

        # X Padding
        self.paddingX_textbox.observe(self.update_x_padding, names="value")

        # Y Padding
        self.paddingY_textbox.observe(self.update_y_padding, names="value")

        # Extra options
        self.extra_options_textbox.observe(self.update_extra_options, names="value")

    def set_observes_obj_specific(self):

        # -- Set observes only for alignment ----------------------------------
        self.num_iterations_textbox.observe(self.update_num_iter, names="value")
        self.batch_size.observe(self.update_batch_size, names="value")
        self.start_button.on_click(self.set_options_and_run)
        self.upsample_factor_textbox.observe(self.update_upsample_factor, names="value")
        
    def make_tab(self):

        # -- Saving ----------------------------------------------------------- 
        save_hbox = HBox(
            self.save_opts_checkboxes,
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        self.save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Save Options",),
        )

        # -- Methods ---------------------------------------------------------- 
        recon_method_box = VBox(self.methods_checkboxes, layout=widgets.Layout(
                    flex_flow="row wrap")
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
                VBox(
                    [
                        HBox(
                            [
                                self.num_iterations_textbox,
                                self.center_of_rotation,
                                self.upsample_factor_textbox,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        HBox(
                            [
                                self.batch_size,
                                self.paddingX_textbox,
                                self.paddingY_textbox,
                                self.downsample_checkbox,
                                self.downsample_factor_textbox,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        self.extra_options_textbox,
                    ],
                    layout=Layout(width="100%", height="100%"),
                )
            ],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Alignment Options",),
        )

        progress_vbox = VBox(
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
                self.methods_accordion,
                self.save_options_accordion,
                self.options_accordion,
                start_button_hb,
                HBox(
                    [progress_vbox, self.plot_output1, self.plot_output2],
                    layout=Layout(flex_wrap="wrap", justify_content="center"),
                ),
            ]
        )


class Recon(Align):

    def __init__(self, Import):

        super().init_attributes(Import)
        self.save_opts_list = ["tomo_before", "recon", "tiff", 
                        "npy"]
        self.widget_type = "Recon"
        init_widgets(self) # initialization of many widgets/some attributes
        super().set_metadata()
        super().set_observes()
        self.set_observes_obj_specific()
        self.make_tab()


    # -- Observe functions for reconstruction ---------------------------------

    # Start button
    def set_options_and_run(self, change):
        change.icon = "fas fa-cog fa-spin fa-lg"
        change.description = (
            "Setting options and loading data into reconstruction algorithm(s)."
        )
        try:
            from tomopyui.backend.tomorecon import TomoRecon
            a = TomoRecon(self)
            change.button_style = "success"
            change.icon = "fa-check-square"
            change.description = "Finished alignment."
        except:
            change.button_style = "warning"
            change.icon = "exclamation-triangle"
            change.description = "Something went wrong."

    # Batch size
    def update_batch_size(self, change):
        self.batch_size = change.new
        self.set_metadata()

    # Number of iterations
    def update_num_iter(self, change):
        self.num_iter = change.new
        self.set_metadata()

    def set_observes_obj_specific(self):

        self.start_button.on_click(self.set_options_and_run)
        self.batch_size.observe(self.update_batch_size, names="value")
        self.num_iterations_textbox.observe(self.update_num_iter, names="value")

    # -- Create recon tab -----------------------------------------------------
    def make_tab(self):

        # -- Saving ----------------------------------------------------------- 
        save_hbox = HBox(
            self.save_opts_checkboxes,
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        self.save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Save Options",),
        )

        # -- Methods ---------------------------------------------------------- 
        recon_method_box = VBox(self.methods_checkboxes, layout=widgets.Layout(
                    flex_flow="row wrap")
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
                VBox(
                    [
                        HBox(
                            [
                                self.num_iterations_textbox,
                                self.center_of_rotation,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        HBox(
                            [
                                self.batch_size,
                                self.paddingX_textbox,
                                self.paddingY_textbox,
                                self.downsample_checkbox,
                                self.downsample_factor_textbox,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        self.extra_options_textbox,
                    ],
                    layout=Layout(width="100%", height="100%"),
                )
            ],
            selected_index=None,
            layout=Layout(width="100%"),
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