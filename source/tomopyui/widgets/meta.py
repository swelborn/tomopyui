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
from ipywidgets import *
import glob
from ._shared.debouncer import debounce
from ._shared import output_handler
import json

# for alignment box
import tomopyui.backend.tomodata as td
from tomopyui.backend.tomoalign import TomoAlign

import logging


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
        self.log_handler, self.log = output_handler.return_handler(self.log,
            logging_level=20)
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
            "rotate": self.rotate
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

        def create_checkbox(description, disabled=False, value=0):
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

        extend_description_style = {"description_width": "auto"}

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
                functools.partial(angle_callbacks, key=metadatakey), names="value",
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
        self.Import =  Import
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
        self.progress_reprojection = IntProgress(description="Reproj: ",value=0, min=0, max=1)
        self.progress_phase_cross_corr = IntProgress(description="Phase Corr: ",value=0, min=0, max=1)
        self.progress_shifting = IntProgress(description="Shifting: ",value=0, min=0, max=1)
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

        extend_description_style = {"description_width": "auto"}

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
                a = TomoAlign(self)
                change.button_style = "success"
                change.icon = "fa-check-square"
                change.description = "Finished alignment."
            except:
                with self.output0:
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
        self.projection_range_x_slider.observe(projection_range_x_update_dict, names="value")

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
        self.projection_range_y_slider.observe(projection_range_y_update_dict, names="value")

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
                Checkbox(description=opt, style=extend_description_style,)
                for opt in opts
            ]
            return checkboxes

        save_checkboxes = create_save_checkboxes(self.accepted_save_opts)

        list(
            (
                opt.observe(
                    functools.partial(
                        create_save_dict_on_checkmark, opt_list=save_checkboxes,
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

        recon_FP_CUDA = Checkbox(description="FP_CUDA")
        recon_BP_CUDA = Checkbox(description="BP_CUDA")
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
            recon_FP_CUDA,
            recon_BP_CUDA,
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
            if change.new == 1:
                self.metadata["opts"]["downsample"] = True
                self.metadata["opts"][
                    "downsample_factor"
                ] = downsample_factor_text.value
                self.downsample = True
                self.downsample_factor = downsample_factor_text.value
                downsample_factor_text.disabled = False
            if change.new == 0:
                self.metadata["opts"]["downsample"] = False
                self.metadata["opts"]["downsample_factor"] = 1
                self.downsample = False
                self.downsample_factor = 1
                downsample_factor_text.value = 1
                downsample_factor_text.disabled = True

        downsample_checkbox = Checkbox(description="Downsample?", value=0)
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

        progress_vbox = VBox([self.progress_total,
                self.progress_reprojection,
                self.progress_phase_cross_corr,
                self.progress_shifting,])

        self.alignment_tab = VBox(
            children=[
                hb1,
                methods_accordion,
                save_options_accordion,
                options_accordion,
                hb2,
                HBox([progress_vbox,
                self.plot_output1, self.plot_output2], layout=Layout(flex_wrap="wrap",
                    justify_content="center"))
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

        extend_description_style = {"description_width": "auto"}
        fpath = self.Import.fpath
        fname = self.Import.fname

        def activate_box(change):
            if change.new == 0:
                radio_recon_fulldataset.disabled = False
                self.metadata["reconstruct"] = True
                self.metadata["opts"] = {}
                self.metadata["methods"] = {}
                self.metadata["save_opts"] = {}
                save_options_accordion.selected_index = 0
                options_accordion.selected_index = 0
                methods_accordion.selected_index = 0
            elif change.new == 1:
                radio_recon_fulldataset.disabled = True
                self.projection_range_x_slider.disabled = True
                self.projection_range_y_slider.disabled = True
                self.metadata["reconstruct"] = False
                self.metadata.pop("opts")
                self.metadata.pop("methods")
                self.metadata.pop("save_opts")
                save_options_accordion.selected_index = None
                options_accordion.selected_index = None
                methods_accordion.selected_index = None

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
            tiff_count_in_folder = len(glob.glob1(fpath, "*.tif"))
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
            os.chdir(fpath)
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
                Checkbox(description=opt, style=extend_description_style,)
                for opt in opts
            ]
            return checkboxes

        save_checkboxes = create_save_checkboxes(save_opts)

        list(
            (
                opt.observe(
                    functools.partial(
                        create_save_dict_on_checkmark, opt_list=save_checkboxes,
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

        recon_FP_CUDA = Checkbox(description="FP_CUDA")
        recon_BP_CUDA = Checkbox(description="BP_CUDA")
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
            recon_FP_CUDA,
            recon_BP_CUDA,
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
                VBox(recon_method_list, 
                        layout=widgets.Layout(flex_flow="row wrap")
                    ),
                sirt_hbox,
            ]
        )

        methods_accordion = Accordion(
            children=[recon_method_box], 
            selected_index=None, 
            titles=("Methods",)
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
            if change.new == 1:
                self.metadata["opts"]["downsample"] = True
                self.metadata["opts"][
                    "downsample_factor"
                ] = downsample_factor_text.value
                self.downsample = True
                self.downsample_factor = downsample_factor_text.value
                downsample_factor_text.disabled = False
            if change.new == 0:
                self.metadata["opts"]["downsample"] = False
                self.metadata["opts"]["downsample_factor"] = 1
                self.downsample = False
                self.downsample_factor = 1
                downsample_factor_text.value = 1
                downsample_factor_text.disabled = True

        downsample_checkbox = Checkbox(description="Downsample?", value=0)
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

        #### putting it all together
        sliders_box = VBox(
            [self.projection_range_x_slider, self.projection_range_y_slider,],
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
            ]
        )