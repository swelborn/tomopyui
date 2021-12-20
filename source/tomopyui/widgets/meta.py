#!/usr/bin/env python

from ipywidgets import *
from ipyfilechooser import FileChooser
import tomopy.prep.normalize

import os
import functools

# fix where this is coming from:
import tomopyui.backend.tomodata as td

# for alignment box
import tifffile as tf
import glob
from ._shared.debouncer import debounce
from ._shared import output_handler
import json

# for alignment box
import tomopyui.backend.tomodata as td

import logging

# for center box
from tomopy.recon.rotation import find_center_vo, find_center, find_center_pc
from tomopyui.backend.util.center import write_center

# for plotting
import matplotlib.pyplot as plt
import numpy as np
import requests

from mpl_interactions import hyperslicer, ioff, interactive_hist, zoom_factory

extend_description_style = {"description_width": "auto"}


class Import:
    def __init__(self):

        self.angle_start = -90
        self.angle_end = 90
        self.num_theta = 360
        self.rotate = False
        self.data_drive = None
        self.tomo = None
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.import_options = [
            "rotate",
        ]
        self.opts_checkboxes = self.create_opts_checkboxes()
        self.angles_textboxes = self.create_angles_textboxes()
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.set_fpath)
        self.metadata = {}
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = output_handler.return_handler(
            self.log, logging_level=20
        )
        self.set_wd()
        self.set_metadata()  # init metadata

    def set_metadata(self):
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "opts": {"rotate": self.rotate},
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta,
            "rotate": self.rotate,
        }

    def set_wd(self, wd=None):
        self.wd = wd

    def set_fpath(self):
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_wd(wd=self.fpath)
        self.set_metadata()

    # Creating options checkboxes and registering their callbacks
    def create_opts_checkboxes(self):
        def create_opt_dict_on_checkmark(change, opt_list):
            self.metadata["opts"] = {opt.description: opt.value for opt in opt_list}
            self.rotate = self.metadata["opts"]["rotate"]

        def create_checkbox(description, disabled=False, value=False):
            checkbox = Checkbox(description=description, disabled=disabled, value=value)
            return checkbox

        opts_checkboxes = []
        for opt in self.import_options:
            opts_checkboxes.append(create_checkbox(opt))

        [
            opt.observe(
                functools.partial(create_opt_dict_on_checkmark, opt_list=[opt]),
                names=["value"],
            )
            for opt in opts_checkboxes
        ]

        return opts_checkboxes

    def create_angles_textboxes(self):
        def create_textbox(description, value, metadatakey, int=False):
            def angle_callbacks(change, key):
                self.metadata[key] = change.new
                if key == "angle_start":
                    self.angle_start = self.metadata[key]
                if key == "angle_end":
                    self.angle_end = self.metadata[key]
                if key == "num_theta":
                    self.num_theta = self.metadata[key]

            if int:
                textbox = IntText(
                    value=value,
                    description=description,
                    disabled=False,
                    style=extend_description_style,
                )

            else:
                textbox = FloatText(
                    value=value,
                    description=description,
                    disabled=False,
                    style=extend_description_style,
                )

            textbox.observe(
                functools.partial(angle_callbacks, key=metadatakey),
                names="value",
            )
            return textbox

        angle_start = create_textbox("Starting angle (\u00b0): ", -90, "angle_start")
        angle_end = create_textbox("Ending angle (\u00b0): ", 90, "angle_end")
        num_projections = create_textbox(
            "Number of Images: ", 360, "num_theta", int=True
        )

        angles_textboxes = [angle_start, angle_end, num_projections]
        return angles_textboxes

    def make_tomo(self):
        self.tomo = td.TomoData(metadata=self.metadata)


class Plotter:
    def __init__(self, Import):

        self.Import = Import
        self.projection_range_x_slider = IntRangeSlider(
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
        self.projection_range_y_slider = IntRangeSlider(
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

    # some stuff here for linking ranges, will become important later:
    # def load_range_from_above_onclick(self):
    #     if self.button_style == "info":
    #         global range_y_link, range_x_link
    #         range_y_link = link(
    #             (projection_range_y_movie, "value"),
    #             (projection_range_y_alignment, "value"),
    #         )
    #         range_x_link = link(
    #             (projection_range_x_movie, "value"),
    #             (projection_range_x_alignment, "value"),
    #         )
    #         self.button_style = "success"
    #         self.description = "Linked ranges. Click again to unlink."
    #         self.icon = "link"
    #     elif self.button_style == "success":
    #         range_y_link.unlink()
    #         range_x_link.unlink()
    #         projection_range_x_alignment.value = [0, tomo.prj_imgs.shape[2] - 1]
    #         projection_range_y_alignment.value = [0, tomo.prj_imgs.shape[1] - 1]
    #         self.button_style = "info"
    #         self.description = "Unlinked ranges. Click again to link."
    #         self.icon = "unlink"

    def create_slicer_with_hist(
        self, plot_type="projection", imagestack=None, Center=None
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
        if plot_type == "projection":
            imagestack = self.Import.tomo.prj_imgs
            theta = self.Import.tomo.theta
            slider_linsp = theta * 180 / np.pi
            slider_str = "Angle:"
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
            self.projection_range_x_slider.value = xlim
            self.projection_range_y_slider.value = ylim[::-1]
            self.set_range_button.button_style = "success"
            self.set_range_button.icon = "square-check"

        self.set_range_button.on_click(set_img_lims_on_click)

        # saving some things in the object
        self.slicer_with_hist_fig = fig
        self.threshold_control = threshold_control
        self.threshold_control_list = [
            slider for slider in threshold_control.vbox.children
        ]

    def save_projection_animation(self):
        """
        Creates button to save animation.
        """

        def save_animation_on_click(click):
            os.chdir(self.Import.fpath)
            self.save_projection_animation_button.button_style = "info"
            self.save_projection_animation_button.icon = "fas fa-cog fa-spin fa-lg"
            self.save_projection_animation_button.description = "Making a movie."
            anim = self.threshold_control.save_animation(
                "projections_animation.mp4",
                self.slicer_with_hist_fig,
                "Angle",
                interval=35,
            )
            self.save_projection_animation_button.button_style = "success"
            self.save_projection_animation_button.icon = "square-check"
            self.save_projection_animation_button.description = (
                "Click again to save another animation."
            )

        self.save_projection_animation_button = Button(
            description="Click to save this animation", layout=Layout(width="auto")
        )

        self.save_projection_animation_button.on_click(save_animation_on_click)


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


class Align:
    def __init__(self, Import, Prep=None, Recon=None):

        # if Prep is None and Recon is None:
        #     self.tomo = Import.tomo
        # if Prep is not None and Align is None:
        #     self.tomo = Prep.tomo
        # if Prep is not None and Align is not None:
        #     self.tomo = Align.tomo
        self.Import = Import
        self.angle_start = Import.angle_start
        self.angle_end = Import.angle_end
        self.num_theta = Import.num_theta
        self.wd = Import.wd
        self.log_handler, self.log = Import.log_handler, Import.log
        self.downsample = False
        self.downsample_factor = 0.5
        self.num_iter = 1
        self.center = 0
        self.upsample_factor = 1
        self.extra_options = {}
        self.batch_size = 20
        self.prj_range_x = (0, 10)
        self.prj_range_y = (0, 10)
        self.paddingX = 10
        self.paddingY = 10
        self.partial = False
        self.methods = {}
        self.save_opts = {}
        self.metadata = {}
        self.accepted_save_opts = ["tomo_after", "tomo_before", "recon", "tiff", "npy"]
        self.set_metadata()

        self.alignbool = False
        self.progress_total = IntProgress(description="Recon: ", value=0, min=0, max=1)
        self.progress_reprojection = IntProgress(
            description="Reproj: ", value=0, min=0, max=1
        )
        self.progress_phase_cross_corr = IntProgress(
            description="Phase Corr: ", value=0, min=0, max=1
        )
        self.progress_shifting = IntProgress(
            description="Shifting: ", value=0, min=0, max=1
        )
        self.plot_output1 = Output()
        self.plot_output2 = Output()

        # self.metadata["callbacks"]["methodoutput"] = method_output
        # self.metadata["callbacks"]["output0"] = output0
        # self.metadata["callbacks"]["output1"] = output1
        # self.metadata["callbacks"]["output2"] = output2
        self.make_alignment_tab()

    # def update_import_metadata(self):
    #     self.angle_start = self.Import.angle_start
    #     self.angle_end = self.Import.angle_end
    #     self.num_theta = self.Import.num_theta
    #     self.fname = self.Import.fname
    #     self.fpath = self.Import.fpath
    #     self.ftype = self.Import.ftype
    #     self.wd = self.Import.wd
    #     self.rotate = self.Import.rotate
    #     self.set_metadata()

    def set_metadata(self):

        self.metadata["opts"] = {}
        self.metadata["opts"]["downsample"] = self.downsample
        self.metadata["opts"]["downsample_factor"] = self.downsample_factor
        self.metadata["opts"]["num_iter"] = self.num_iter
        self.metadata["opts"]["center"] = self.center
        self.metadata["opts"]["upsample_factor"] = self.upsample_factor
        self.metadata["opts"]["batch_size"] = self.batch_size
        self.metadata["opts"]["pad"] = (
            self.paddingX,
            self.paddingY,
        )
        self.metadata["opts"]["extra_options"] = self.extra_options
        self.metadata["methods"] = self.methods
        self.metadata["save_opts"] = self.save_opts
        self.metadata["prj_range_x"] = self.prj_range_x
        self.metadata["prj_range_y"] = self.prj_range_y
        self.metadata["partial"] = self.partial
        self.log.info(f"Set alignment metadata to {self.metadata}")

    def make_alignment_tab(self):
        def activate_box(change):
            if change.new == 0:
                radio_align_fulldataset.disabled = False
                self.alignbool = True
                save_options_accordion.selected_index = 0
                options_accordion.selected_index = 0
                methods_accordion.selected_index = 0
                align_start_button.disabled = False
                self.log.info("Activated alignment.")
            elif change.new == 1:
                radio_align_fulldataset.disabled = True
                self.projection_range_x_slider.disabled = True
                self.projection_range_y_slider.disabled = True
                self.alignbool = False
                save_options_accordion.selected_index = None
                options_accordion.selected_index = None
                methods_accordion.selected_index = None
                align_start_button.disabled = True
                self.log.info("Deactivated alignment.")

        def set_projection_ranges(sizeY, sizeX):
            self.projection_range_x_slider.max = sizeX - 1
            self.projection_range_y_slider.max = sizeY - 1
            # projection_range_z_recon.max = sizeZ-1
            self.projection_range_x_slider.value = [0, sizeX - 1]
            self.projection_range_y_slider.value = [0, sizeY - 1]
            # projection_range_z_recon.value = [0, sizeZ-1]
            self.metadata["prj_range_x"] = self.projection_range_x_slider.value
            self.metadata["prj_range_y"] = self.projection_range_y_slider.value
            self.prj_range_x = self.metadata["prj_range_x"]
            self.prj_range_y = self.metadata["prj_range_y"]

        def load_tif_shape_tag(folder_import=False):
            os.chdir(self.Import.fpath)
            tiff_count_in_folder = len(glob.glob1(self.Import.fpath, "*.tif"))
            global sizeY, sizeX
            if folder_import:
                _tomo = td.TomoData(metadata=self.metadata)
                size = _tomo.prj_imgs.shape
                # sizeZ = size[0]
                sizeY = size[1]
                sizeX = size[2]
                set_projection_ranges(sizeY, sizeX)
            else:
                with tf.TiffFile(self.Import.fname) as tif:
                    if tiff_count_in_folder > 50:
                        sizeX = tif.pages[0].tags["ImageWidth"].value
                        sizeY = tif.pages[0].tags["ImageLength"].value
                        # sizeZ = tiff_count_in_folder # can maybe use this later
                    else:
                        imagesize = tif.pages[0].tags["ImageDescription"]
                        size = json.loads(imagesize.value)["shape"]
                        sizeY = size[1]
                        sizeX = size[2]
                    set_projection_ranges(sizeY, sizeX)

        def load_npy_shape():
            os.chdir(self.Import.fpath)
            size = np.load(self.Import.fname, mmap_mode="r").shape
            global sizeY, sizeX
            sizeY = size[1]
            sizeX = size[2]
            set_projection_ranges(sizeY, sizeX)

        def activate_full_partial(change):
            if change.new == 1:
                self.metadata["partial"] = True
                self.projection_range_x_slider.disabled = False
                self.projection_range_y_slider.disabled = False
                # projection_range_z_recon.disabled = False
                if self.Import.fname != "":
                    if self.Import.fname.__contains__(".tif"):
                        load_tif_shape_tag()
                    elif self.Import.fname.__contains__(".npy"):
                        load_npy_shape()
                    else:
                        load_tif_shape_tag(folder_import=True)
            elif change.new == 0:
                self.metadata["partial"] = False
                set_projection_ranges(sizeY, sizeX)
                self.projection_range_x_slider.disabled = True
                self.projection_range_y_slider.disabled = True
                # projection_range_z_recon.disabled = True

        def set_options_and_run_align(change):
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

        radio_align = RadioButtons(
            options=["Yes", "No"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            value="No",
        )
        radio_align.observe(activate_box, names="index")

        radio_align_fulldataset = RadioButtons(
            options=["Full", "Partial"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            disabled=True,
            value="Full",
        )
        radio_align_fulldataset.observe(activate_full_partial, names="index")

        @debounce(0.2)
        def projection_range_x_update_dict(change):
            self.metadata["prj_range_x"] = change.new

        self.projection_range_x_slider = IntRangeSlider(
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
        self.projection_range_x_slider.observe(
            projection_range_x_update_dict, names="value"
        )

        @debounce(0.2)
        def projection_range_y_update_dict(change):
            self.metadata["prj_range_y"] = change.new

        self.projection_range_y_slider = IntRangeSlider(
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
        self.projection_range_y_slider.observe(
            projection_range_y_update_dict, names="value"
        )

        # Radio descriptions
        radio_description = "Would you like to align this dataset?"
        partial_radio_description = (
            "Would you like to use the full dataset, or a partial dataset?"
        )
        radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + radio_description
            + " </p>"
        )
        partial_radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + partial_radio_description
            + " </p>"
        )
        # --------------------------------------------------------------- Saving
        # Saving options
        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_save_dict_on_checkmark(change, opt_list):
            self.metadata["save_opts"] = create_option_dictionary(opt_list)
            self.save_opts = self.metadata["save_opts"]

        self.metadata["save_opts"] = {key: None for key in self.accepted_save_opts}
        self.save_opts = self.metadata["save_opts"]

        def create_save_checkboxes(opts):
            checkboxes = [
                Checkbox(
                    description=opt,
                    style=extend_description_style,
                )
                for opt in opts
            ]
            return checkboxes

        save_checkboxes = create_save_checkboxes(self.accepted_save_opts)

        list(
            (
                opt.observe(
                    functools.partial(
                        create_save_dict_on_checkmark,
                        opt_list=save_checkboxes,
                    ),
                    names=["value"],
                )
                for opt in save_checkboxes
            )
        )

        save_hbox = HBox(
            save_checkboxes,
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Save Options",),
        )
        # -------------------------------------------------------------- Methods
        # Methods checkboxes
        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_dict_on_checkmark(change, opt_list, dictname):
            self.metadata["methods"][dictname] = create_option_dictionary(opt_list)

        def create_dict_on_checkmark_no_options(change):
            if change.new == True:
                self.metadata["methods"][change.owner.description] = {}
            if change.new == False:
                self.metadata["methods"].pop(change.owner.description)

        recon_FBP_CUDA = Checkbox(description="FBP_CUDA")
        ### !!!!!!!! sirt cuda has options - maybe make them into a radio chooser
        recon_SIRT_CUDA = Checkbox(description="SIRT_CUDA")
        recon_SIRT_CUDA_option1 = Checkbox(
            description="SIRT Plugin-Faster", disabled=False
        )
        recon_SIRT_CUDA_option2 = Checkbox(
            description="SIRT 3D-Fastest", disabled=False
        )
        recon_SIRT_CUDA_option_list = [
            recon_SIRT_CUDA_option1,
            recon_SIRT_CUDA_option2,
        ]
        recon_SIRT_CUDA_checkboxes = [
            recon_SIRT_CUDA,
            recon_SIRT_CUDA_option1,
            recon_SIRT_CUDA_option2,
        ]
        recon_SART_CUDA = Checkbox(description="SART_CUDA")
        recon_CGLS_CUDA = Checkbox(description="CGLS_CUDA")
        recon_MLEM_CUDA = Checkbox(description="MLEM_CUDA")
        recon_method_list = [
            recon_FBP_CUDA,
            recon_SART_CUDA,
            recon_CGLS_CUDA,
            recon_MLEM_CUDA,
        ]
        [
            checkbox.observe(create_dict_on_checkmark_no_options)
            for checkbox in recon_method_list
        ]

        # Toggling on other options if you select SIRT. Better to use radio here.
        def toggle_on(change, opt_list, dictname):
            if change.new == True:
                self.metadata["methods"][dictname] = {}
                for option in opt_list:
                    option.disabled = False
            if change.new == False:
                self.metadata["methods"].pop(dictname)
                for option in opt_list:
                    option.value = False
                    option.disabled = True

        recon_SIRT_CUDA.observe(
            functools.partial(
                toggle_on, opt_list=recon_SIRT_CUDA_option_list, dictname="SIRT_CUDA"
            ),
            names=["value"],
        )

        # Maps options to observe functions.
        # If other options needed for other reconstruction methods, use similar
        list(
            (
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark,
                        opt_list=recon_SIRT_CUDA_option_list,
                        dictname="SIRT_CUDA",
                    ),
                    names=["value"],
                )
                for opt in recon_SIRT_CUDA_option_list
            )
        )

        sirt_hbox = HBox(recon_SIRT_CUDA_checkboxes)
        recon_method_box = VBox(
            [
                VBox(recon_method_list, layout=widgets.Layout(flex_flow="row wrap")),
                sirt_hbox,
            ]
        )
        methods_accordion = Accordion(
            children=[recon_method_box], selected_index=None, titles=("Methods",)
        )
        # -------------------------------------------------------------- Options
        # Number of iterations
        def update_num_iter_dict(change):
            self.metadata["opts"]["num_iter"] = change.new
            self.num_iter = change.new
            self.progress_total.max = change.new

        number_of_align_iterations = IntText(
            description="Number of Iterations: ",
            style=extend_description_style,
            value=20,
        )
        number_of_align_iterations.observe(update_num_iter_dict, names="value")

        # Center of rotation
        def update_center_of_rotation_dict(change):
            self.metadata["opts"]["center"] = change.new
            self.center = change.new

        center_of_rotation = FloatText(
            description="Center of Rotation: ",
            style=extend_description_style,
            value=self.metadata["opts"]["center"],
        )
        center_of_rotation.observe(update_center_of_rotation_dict, names="value")

        # Downsampling
        def downsample_turn_on(change):
            if change.new == True:
                self.metadata["opts"]["downsample"] = True
                self.metadata["opts"][
                    "downsample_factor"
                ] = downsample_factor_text.value
                self.downsample = True
                self.downsample_factor = downsample_factor_text.value
                downsample_factor_text.disabled = False
            if change.new == False:
                self.metadata["opts"]["downsample"] = False
                self.metadata["opts"]["downsample_factor"] = 1
                self.downsample = False
                self.downsample_factor = 1
                downsample_factor_text.value = 1
                downsample_factor_text.disabled = True

        downsample_checkbox = Checkbox(description="Downsample?", value=False)
        downsample_checkbox.observe(downsample_turn_on)

        def update_downsample_factor_dict(change):
            self.metadata["opts"]["downsample_factor"] = change.new
            self.downsample_factor = change.new

        downsample_factor_text = BoundedFloatText(
            value=1,
            min=0.001,
            max=1.0,
            description="Downsampe factor:",
            disabled=True,
            style=extend_description_style,
        )
        downsample_factor_text.observe(update_downsample_factor_dict, names="value")

        # Upsample factor
        def update_upsample_factor_dict(change):
            self.metadata["opts"]["upsample_factor"] = change.new
            self.upsample_factor = change.new

        upsample_factor = FloatText(
            description="Upsample Factor: ",
            style=extend_description_style,
            value=self.metadata["opts"]["upsample_factor"],
        )
        upsample_factor.observe(update_upsample_factor_dict, names="value")

        # Batch size
        def update_batch_size_dict(change):
            self.metadata["opts"]["batch_size"] = change.new
            self.batch_size = change.new
            self.progress_phase_cross_corr.max = change.new
            self.progress_shifting.max = change.new
            self.progress_reprojection.max = change.new

        batch_size = IntText(
            description="Batch size (for GPU): ",
            style=extend_description_style,
            value=self.metadata["opts"]["batch_size"],
        )
        batch_size.observe(update_batch_size_dict, names="value")

        # X Padding
        def update_x_padding_dict(change):
            self.paddingX = change.new
            self.metadata["opts"]["pad"] = (
                self.paddingX,
                self.paddingY,
            )
            self.pad = self.metadata["opts"]["pad"]

        paddingX = IntText(
            description="Padding X (px): ",
            style=extend_description_style,
            value=self.paddingX,
        )
        paddingX.observe(update_x_padding_dict, names="value")

        # Y Padding
        def update_y_padding_dict(change):
            self.paddingY = change.new
            self.metadata["opts"]["pad"] = (
                self.paddingX,
                self.paddingY,
            )
            self.pad = self.metadata["opts"]["pad"]

        paddingY = IntText(
            description="Padding Y (px): ",
            style=extend_description_style,
            value=self.paddingY,
        )
        paddingY.observe(update_x_padding_dict, names="value")

        def update_extra_options_dict(change):
            self.self.metadatadata["opts"]["extra_options"] = change.new

        extra_options = Text(
            description="Extra options: ",
            placeholder='{"MinConstraint": 0}',
            style=extend_description_style,
        )
        extra_options.observe(update_extra_options_dict, names="value")

        align_start_button = Button(
            description="After choosing all of the options above, click this button to start the alignment.",
            disabled=True,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start aligning things with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        align_start_button.on_click(set_options_and_run_align)

        # ----------------------------------------------------- Box organization
        radio_description = (
            "Would you like to try automatic alignment before reconstruction?"
        )
        partial_radio_description = (
            "Would you like to use the full dataset, or a partial dataset?"
        )
        radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + radio_description
            + " </p>"
        )
        partial_radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + partial_radio_description
            + " </p>"
        )

        pixel_range_slider_vb = VBox(
            [
                # HBox(
                #     [load_range_from_above],
                #     justify_content="center",
                #     align_content="center",
                # ),
                self.projection_range_x_slider,
                self.projection_range_y_slider,
            ],
            layout=Layout(width="30%"),
            justify_content="center",
            align_items="space-between",
        )

        hb1 = HBox(
            [
                radio_description,
                radio_align,
                partial_radio_description,
                radio_align_fulldataset,
                pixel_range_slider_vb,
            ],
            layout=Layout(
                width="auto",
                justify_content="center",
                align_items="center",
                flex="flex-grow",
            ),
        )
        hb2 = HBox(
            [align_start_button], layout=Layout(width="auto", justify_content="center")
        )

        options_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [
                                number_of_align_iterations,
                                center_of_rotation,
                                upsample_factor,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        HBox(
                            [
                                batch_size,
                                paddingX,
                                paddingY,
                                downsample_checkbox,
                                downsample_factor_text,
                            ],
                            layout=Layout(
                                flex_wrap="wrap", justify_content="space-between"
                            ),
                        ),
                        extra_options,
                    ],
                    layout=Layout(width="100%", height="100%"),
                )
            ],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Other Alignment Options",),
        )

        progress_vbox = VBox(
            [
                self.progress_total,
                self.progress_reprojection,
                self.progress_phase_cross_corr,
                self.progress_shifting,
            ]
        )

        self.alignment_tab = VBox(
            children=[
                hb1,
                methods_accordion,
                save_options_accordion,
                options_accordion,
                hb2,
                HBox(
                    [progress_vbox, self.plot_output1, self.plot_output2],
                    layout=Layout(flex_wrap="wrap", justify_content="center"),
                ),
            ]
        )


# class Plotter:

#     def __init__(self, Import=None, Center=None, Prep=None, Align=None, Recon=None):

#         self.Import = Import
#         self.Center = Center
#         self.Prep = Prep
#         self.Align = Align
#         self.Recon = Recon
#         self.centers_figure = None
#         self.recon_figure = None

#     def plot_centers(self):

#         def plot_projections(tomodata, range_x, range_y, range_z, skip, scale_factor):
#             volume = tomodata.prj_imgs[
#                 range_z[0] : range_z[1] : skip,
#                 range_y[0] : range_y[1] : 1,
#                 range_x[0] : range_x[1] : 1,
#             ].copy()
#             volume_rescaled = rescale(
#                 volume, (1, scale_factor, scale_factor), anti_aliasing=False
#             )
#             fig = px.imshow(
#                 volume_rescaled,
#                 facet_col=0,
#                 facet_col_wrap=5,
#                 binary_string=True,
#                 height=2000,
#                 facet_row_spacing=0.001,
#             )
#             rangez = range(range_z[0], range_z[1], skip)
#             angles = np.around(tomodata.theta * 180 / np.pi, decimals=1)
#             for i, j in enumerate(rangez):
#                 for k in range(len(rangez)):
#                     if fig.layout.annotations[k]["text"] == "facet_col=" + str(i):
#                         fig.layout.annotations[k]["text"] = (
#                             "Proj:" + " " + str(j) + "<br>Angle:" + "" + str(angles[j])
#                         )
#                         fig.layout.annotations[k]["y"] = (
#                             fig.layout.annotations[k]["y"] - 0.02
#                         )
#                         break
#             fig.update_layout(
#                 # margin=dict(autoexpand=False),
#                 font_family="Helvetica",
#                 font_size=30,
#                 # margin=dict(l=5, r=5, t=5, b=5),
#                 paper_bgcolor="LightSteelBlue",
#             )
#             fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#             # fig.for_each_annotation(lambda a: a.update(text=''))
#             display(fig)

#     def plot_imported_data(self):


#         plot_output = Output()
#         movie_output = Output()
#         from matplotlib import animation
#         from matplotlib import pyplot as plt
#         from IPython.display import HTML
#         import plotly.express as px
#         from skimage.transform import rescale

#         def plot_projections(tomodata, range_x, range_y, range_z, skip, scale_factor):
#             volume = tomodata.prj_imgs[
#                 range_z[0] : range_z[1] : skip,
#                 range_y[0] : range_y[1] : 1,
#                 range_x[0] : range_x[1] : 1,
#             ].copy()
#             volume_rescaled = rescale(
#                 volume, (1, scale_factor, scale_factor), anti_aliasing=False
#             )
#             fig = px.imshow(
#                 volume_rescaled,
#                 facet_col=0,
#                 facet_col_wrap=5,
#                 binary_string=True,
#                 height=2000,
#                 facet_row_spacing=0.001,
#             )
#             rangez = range(range_z[0], range_z[1], skip)
#             angles = np.around(tomodata.theta * 180 / np.pi, decimals=1)
#             for i, j in enumerate(rangez):
#                 for k in range(len(rangez)):
#                     if fig.layout.annotations[k]["text"] == "facet_col=" + str(i):
#                         fig.layout.annotations[k]["text"] = (
#                             "Proj:" + " " + str(j) + "<br>Angle:" + "" + str(angles[j])
#                         )
#                         fig.layout.annotations[k]["y"] = (
#                             fig.layout.annotations[k]["y"] - 0.02
#                         )
#                         break
#             fig.update_layout(
#                 # margin=dict(autoexpand=False),
#                 font_family="Helvetica",
#                 font_size=30,
#                 # margin=dict(l=5, r=5, t=5, b=5),
#                 paper_bgcolor="LightSteelBlue",
#             )
#             fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#             # fig.for_each_annotation(lambda a: a.update(text=''))
#             display(fig)


class Center:
    def __init__(self, Import, Prep=None, Align=None, Recon=None):
        self.Import = Import
        self.Prep = Prep
        self.Align = Align
        self.Recon = Recon
        # add a method in Import projection range (just like in Align/Prep)
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
            self.center_plotter.set_range_button.icon = "question"
            self.center_plotter.set_range_button.description = ""

            # Make VBox instantiated outside into the plot
            self.center_tab.children[2].children[0].children[2].children = [
                HBox([self.center_plotter.slicer_with_hist_fig.canvas]),
                HBox(self.center_plotter.threshold_control_list),
                HBox([self.center_plotter.set_range_button]),
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
            tooltip="Start reconstruction with this button.",
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


class Recon:
    def __init__(self, Import, Prep=None, Align=None):

        if Prep is None and Align is None:
            self.tomo = Import.tomo
        if Prep is not None and Align is None:
            self.tomo = Prep.tomo
        if Prep is not None and Align is not None:
            self.tomo = Align.tomo
        self.Import = Import
        self.metadata = Import.metadata.copy()
        self.opts = {}
        self.methods = {}
        self.save_opts = {}
        self.downsample = False
        self.downsample_factor = 1
        self.num_iter = 1
        self.partial = False
        self.center = 0
        self.log_handler, self.log = Import.log_handler, Import.log
        self.prj_range_x = (0, 10)
        self.prj_range_y = (0, 10)
        self.reconbool = False
        self.set_metadata()
        self.make_recon_tab()

    def set_metadata(self):
        self.metadata["opts"] = self.opts
        self.metadata["methods"] = self.methods
        self.metadata["save_opts"] = self.save_opts
        self.metadata["opts"]["downsample"] = self.downsample
        self.metadata["opts"]["downsample_factor"] = self.downsample_factor
        self.metadata["partial"] = self.partial

    def make_recon_tab(self):
        def activate_box(change):
            if change.new == 0:
                radio_recon_fulldataset.disabled = False
                self.metadata["reconstruct"] = True
                save_options_accordion.selected_index = 0
                options_accordion.selected_index = 0
                methods_accordion.selected_index = 0
                recon_start_button.disabled = False
                self.log.info("Activated reconstruction.")
            elif change.new == 1:
                radio_recon_fulldataset.disabled = True
                self.projection_range_x_slider.disabled = True
                self.projection_range_y_slider.disabled = True
                self.metadata["reconstruct"] = False
                save_options_accordion.selected_index = None
                options_accordion.selected_index = None
                methods_accordion.selected_index = None
                recon_start_button.disabled = True
                self.log.info("Deactivated reconstruction.")

        def set_projection_ranges(sizeY, sizeX):
            self.projection_range_x_slider.max = sizeX - 1
            self.projection_range_y_slider.max = sizeY - 1
            # projection_range_z_recon.max = sizeZ-1
            self.projection_range_x_slider.value = [0, sizeX - 1]
            self.projection_range_y_slider.value = [0, sizeY - 1]
            # projection_range_z_recon.value = [0, sizeZ-1]
            self.metadata["prj_range_x"] = self.projection_range_x_slider.value
            self.metadata["prj_range_y"] = self.projection_range_y_slider.value
            self.prj_range_x = self.metadata["prj_range_x"]
            self.prj_range_y = self.metadata["prj_range_y"]

        def load_tif_shape_tag(folder_import=False):
            os.chdir(self.Import.fpath)
            tiff_count_in_folder = len(glob.glob1(self.Import.fpath, "*.tif"))
            global sizeY, sizeX
            if folder_import:
                _tomo = td.TomoData(metadata=self.metadata)
                size = _tomo.prj_imgs.shape
                # sizeZ = size[0]
                sizeY = size[1]
                sizeX = size[2]
                set_projection_ranges(sizeY, sizeX)

            else:
                with tf.TiffFile(self.Import.fname) as tif:
                    if tiff_count_in_folder > 50:
                        sizeX = tif.pages[0].tags["ImageWidth"].value
                        sizeY = tif.pages[0].tags["ImageLength"].value
                        # sizeZ = tiff_count_in_folder # can maybe use this later
                    else:
                        imagesize = tif.pages[0].tags["ImageDescription"]
                        size = json.loads(imagesize.value)["shape"]
                        sizeY = size[1]
                        sizeX = size[2]
                    set_projection_ranges(sizeY, sizeX)

        def load_npy_shape():
            os.chdir(self.Import.fpath)
            size = np.load(self.Import.fname, mmap_mode="r").shape
            global sizeY, sizeX
            sizeY = size[1]
            sizeX = size[2]
            set_projection_ranges(sizeY, sizeX)

        def activate_full_partial(change):
            if change.new == 1:
                self.metadata["partial"] = True
                self.projection_range_x_slider.disabled = False
                self.projection_range_y_slider.disabled = False
                # projection_range_z_recon.disabled = False
                if self.Import.fname != "":
                    if self.Import.fname.__contains__(".tif"):
                        load_tif_shape_tag()
                    elif self.Import.fname.__contains__(".npy"):
                        load_npy_shape()
                    else:
                        load_tif_shape_tag(folder_import=True)
            elif change.new == 0:
                self.metadata["partial"] = False
                set_projection_ranges(sizeY, sizeX)
                self.projection_range_x_slider.disabled = True
                self.projection_range_y_slider.disabled = True
                # projection_range_z_recon.disabled = True

        radio_recon = RadioButtons(
            options=["Yes", "No"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            value="No",
        )
        radio_recon.observe(activate_box, names="index")

        radio_recon_fulldataset = RadioButtons(
            options=["Full", "Partial"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            disabled=True,
            value="Full",
        )
        radio_recon_fulldataset.observe(activate_full_partial, names="index")

        # Callbacks for projection range sliders
        @debounce(0.2)
        def projection_range_x_update_dict(change):
            self.metadata["prj_range_x"] = change.new

        self.projection_range_x_slider = IntRangeSlider(
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
        self.projection_range_x_slider.observe(projection_range_x_update_dict, "value")

        @debounce(0.2)
        def projection_range_y_update_dict(change):
            self.metadata["prj_range_y"] = change.new

        self.projection_range_y_slider = IntRangeSlider(
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
        self.projection_range_y_slider.observe(projection_range_y_update_dict, "value")

        # Radio descriptions
        radio_description = "Would you like to reconstruct this dataset?"
        partial_radio_description = (
            "Would you like to use the full dataset, or a partial dataset?"
        )
        radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + radio_description
            + " </p>"
        )
        partial_radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + partial_radio_description
            + " </p>"
        )

        # Saving options
        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_save_dict_on_checkmark(change, opt_list):
            self.metadata["save_opts"] = create_option_dictionary(opt_list)

        save_opts = ["tomo_before", "recon", "tiff", "npy"]
        self.metadata["save_opts"] = {key: None for key in save_opts}

        def create_save_checkboxes(opts):
            checkboxes = [
                Checkbox(
                    description=opt,
                    style=extend_description_style,
                )
                for opt in opts
            ]
            return checkboxes

        save_checkboxes = create_save_checkboxes(save_opts)

        list(
            (
                opt.observe(
                    functools.partial(
                        create_save_dict_on_checkmark,
                        opt_list=save_checkboxes,
                    ),
                    names=["value"],
                )
                for opt in save_checkboxes
            )
        )

        save_hbox = HBox(
            save_checkboxes,
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        save_options_accordion = Accordion(
            children=[save_hbox],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Save Options",),
        )

        # Methods checkboxes
        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_dict_on_checkmark(change, opt_list, dictname):
            self.metadata["methods"][dictname] = create_option_dictionary(opt_list)

        def create_dict_on_checkmark_no_options(change):
            if change.new == True:
                self.metadata["methods"][change.owner.description] = {}
            if change.new == False:
                self.metadata["methods"].pop(change.owner.description)

        recon_FBP_CUDA = Checkbox(description="FBP_CUDA")
        ### !!!!!!!! sirt cuda has options - maybe make them into a radio chooser
        recon_SIRT_CUDA = Checkbox(description="SIRT_CUDA")
        recon_SIRT_CUDA_option1 = Checkbox(
            description="SIRT Plugin-Faster", disabled=False
        )
        recon_SIRT_CUDA_option2 = Checkbox(
            description="SIRT 3D-Fastest", disabled=False
        )
        recon_SIRT_CUDA_option_list = [
            recon_SIRT_CUDA_option1,
            recon_SIRT_CUDA_option2,
        ]
        recon_SIRT_CUDA_checkboxes = [
            recon_SIRT_CUDA,
            recon_SIRT_CUDA_option1,
            recon_SIRT_CUDA_option2,
        ]
        recon_SART_CUDA = Checkbox(description="SART_CUDA")
        recon_CGLS_CUDA = Checkbox(description="CGLS_CUDA")
        recon_MLEM_CUDA = Checkbox(description="MLEM_CUDA")
        recon_method_list = [
            recon_FBP_CUDA,
            recon_SART_CUDA,
            recon_CGLS_CUDA,
            recon_MLEM_CUDA,
        ]
        [
            checkbox.observe(create_dict_on_checkmark_no_options)
            for checkbox in recon_method_list
        ]

        # Toggling on other options if you select SIRT. Better to use radio here.
        def toggle_on(change, opt_list, dictname):
            if change.new == 1:
                self.metadata["methods"][dictname] = {}
                for option in opt_list:
                    option.disabled = False
            if change.new == 0:
                self.metadata["methods"].pop(dictname)
                for option in opt_list:
                    option.value = 0
                    option.disabled = True

        recon_SIRT_CUDA.observe(
            functools.partial(
                toggle_on, opt_list=recon_SIRT_CUDA_option_list, dictname="SIRT_CUDA"
            ),
            names=["value"],
        )

        # Maps options to observe functions.
        # If other options needed for other reconstruction methods, use similar
        list(
            (
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark,
                        opt_list=recon_SIRT_CUDA_option_list,
                        dictname="SIRT_CUDA",
                    ),
                    names=["value"],
                )
                for opt in recon_SIRT_CUDA_option_list
            )
        )

        sirt_hbox = HBox(recon_SIRT_CUDA_checkboxes)

        recon_method_box = VBox(
            [
                VBox(recon_method_list, layout=widgets.Layout(flex_flow="row wrap")),
                sirt_hbox,
            ]
        )

        methods_accordion = Accordion(
            children=[recon_method_box], selected_index=None, titles=("Methods",)
        )

        # Options
        # number of iterations
        self.metadata["opts"]["num_iter"] = 20
        self.num_iter = 20

        def update_num_iter_dict(change):
            self.metadata["opts"]["num_iter"] = change.new

        number_of_recon_iterations = IntText(
            description="Number of Iterations: ",
            style=extend_description_style,
            value=20,
        )
        number_of_recon_iterations.observe(update_num_iter_dict, names="value")

        # center of rotation
        self.metadata["opts"]["center"] = 0

        def update_center_of_rotation_dict(change):
            self.metadata["opts"]["center"] = change.new

        center_of_rotation = FloatText(
            description="Center of Rotation: ",
            style=extend_description_style,
            value=self.metadata["opts"]["center"],
        )
        center_of_rotation.observe(update_center_of_rotation_dict, names="value")

        self.metadata["opts"]["extra_options"] = None

        def update_extra_options_dict(change):
            self.self.metadatadata["opts"]["extra_options"] = change.new

        extra_options = Text(
            description="Extra options: ",
            placeholder='{"MinConstraint": 0}',
            style=extend_description_style,
        )
        extra_options.observe(update_extra_options_dict, names="value")

        # Downsampling
        def downsample_turn_on(change):
            if change.new == True:
                self.metadata["opts"]["downsample"] = True
                self.metadata["opts"][
                    "downsample_factor"
                ] = downsample_factor_text.value
                self.downsample = True
                self.downsample_factor = downsample_factor_text.value
                downsample_factor_text.disabled = False
            if change.new == False:
                self.metadata["opts"]["downsample"] = False
                self.metadata["opts"]["downsample_factor"] = 1
                self.downsample = False
                self.downsample_factor = 1
                downsample_factor_text.value = 1
                downsample_factor_text.disabled = True

        downsample_checkbox = Checkbox(description="Downsample?", value=False)
        downsample_checkbox.observe(downsample_turn_on)

        def downsample_factor_update_dict(change):
            self.metadata["opts"]["downsample_factor"] = change.new
            self.downsample_factor = change.new

        downsample_factor_text = BoundedFloatText(
            value=1,
            min=0.001,
            max=1.0,
            description="Downsample factor:",
            disabled=True,
            style=extend_description_style,
        )
        downsample_factor_text.observe(downsample_factor_update_dict, names="value")
        downsample_hb = HBox(
            [downsample_checkbox, downsample_factor_text],
            layout=Layout(flex_wrap="wrap", justify_content="space-between"),
        )

        options_accordion = Accordion(
            children=[
                HBox(
                    [
                        number_of_recon_iterations,
                        center_of_rotation,
                        downsample_checkbox,
                        downsample_factor_text,
                        extra_options,
                    ],
                    layout=Layout(
                        flex_flow="row wrap", justify_content="space-between"
                    ),
                ),
            ],
            selected_index=None,
            layout=Layout(width="100%"),
            titles=("Options",),
        )

        def set_options_and_run_recon(change):
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

        recon_start_button = Button(
            description="After choosing all of the options above, click this button to start the reconstruction.",
            disabled=True,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start reconstruction with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        recon_start_button.on_click(set_options_and_run_recon)

        #### putting it all together
        sliders_box = VBox(
            [
                self.projection_range_x_slider,
                self.projection_range_y_slider,
            ],
            layout=Layout(width="30%"),
            justify_content="center",
            align_items="space-between",
        )

        recon_initialization_box = HBox(
            [
                radio_description,
                radio_recon,
                partial_radio_description,
                radio_recon_fulldataset,
                sliders_box,
            ],
            layout=Layout(
                width="auto",
                justify_content="center",
                align_items="center",
                flex="flex-grow",
            ),
        )

        self.recon_tab = VBox(
            [
                recon_initialization_box,
                options_accordion,
                methods_accordion,
                save_options_accordion,
                recon_start_button,
            ]
        )
