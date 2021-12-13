from ipywidgets import *
from ipyfilechooser import FileChooser
import tomopy.prep.normalize

import os
import functools
# fix where this is coming from:
from .. import tomodata as td

# for alignment box
import tifffile as tf
from ipywidgets import *
import glob
from .debouncer import debounce
import functools
import json


class Import():

    def __init__(self):
        
        self.angle_start = None
        self.angle_end = None
        self.num_theta = None
        self.data_drive = None
        self.tomo = None
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.import_options = ["rotate",]
        self.opts_checkboxes = self.create_opts_checkboxes()
        self.angles_textboxes = self.create_angles_textboxes()
        self.filechooser = FileChooser()
        self.filechooser.register_callback(set_fpath)
        self.metadata = {}
        self.set_metadata() # init metadata
        
    def set_metadata(self):
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "opts": {"rotate" : self.rotate},
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta
        }
        print(self.metadata)
    def set_wd(self, wd=None):
        self.wd = wd
    
    def set_fpath(self):
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_metadata()

    # Creating options checkboxes and registering their callbacks
    def create_opts_checkboxes(self):

        def create_opt_dict_on_checkmark(change, opt_list):
            self.metadata["opts"] = {opt.description: opt.value for opt in opt_list}

        def create_checkbox(description, disabled=False, value=0):
            checkbox = Checkbox(description=description, 
                disabled=disabled, 
                value=value)
            return checkbox

        opts_checkboxes = []
        for opt in self.import_options:
            opts_checkboxes.append(create_checkbox(opt))

        [
            opt.observe(
                functools.partial(
                    create_opt_dict_on_checkmark,
                    opt_list=[opt]
                ),
                names=["value"],
            )
            for opt in opts_checkboxes
        ]

        return opts_checkboxes

    def create_angles_textboxes(self):
        
        extend_description_style = {"description_width": "auto"}
        
        def create_textbox(description, value, metadatakey, int=False):
            
            def angle_callbacks(change, key):
                print("hello")
                self.metadata[key] = change.new

            if int:
                textbox = IntText(
                value=value, 
                description=description, 
                disabled=False,
                style=extend_description_style
                )

            else:
                textbox = FloatText(
                value=value, 
                description=description, 
                disabled=False,
                style=extend_description_style
                )

            textbox.observe(functools.partial(
                angle_callbacks, 
                key=metadatakey),
                names="value",
            )
            return textbox

        angle_start = create_textbox("Starting angle (\u00b0):", -90, "angle_start")
        angle_end = create_textbox("Ending angle (\u00b0):", 90, "angle_end")
        num_projections = create_textbox(
            "Number of Images", 360, "num_theta", int=True 
        )

        angles_textboxes = [angle_start, angle_end, num_projections]
        return angles_textboxes

    def make_tomo(self):
        self.tomo = td.TomoData(metadata=self.metadata)


class Prep():

    def __init__(self, Import):
        self.tomo = Import.tomo
        self.dark = None
        self.flat = None
        self.darkfc = FileChooser()
        self.darkfc.register_callback(set_fpath_dark)
        self.flatfc = FileChooser()
        self.flatfc.register_callback(set_fpath_flat)
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
            "opts": {"rotate" : self.rotate},
        }

    def set_metadata_flat(self):
        self.flatmetadata = {
            "fpath": self.fpathflat,
            "fname": self.fnameflat,
            "opts": {"rotate" : self.rotate},
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
            self.tomo.prj_imgs, 
            self.flat.prj_imgs, 
            self.dark.prj_imgs
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

class Align():

    def __init__(self, Import, Prep=None, Recon=None):
        print("hello")



class Recon():

    def __init__(self, Import, Prep=None, Align=None):

        if Prep is None and Align is None:
            self.tomo = Import.tomo
        if Prep is not None and Align is None:
            self.tomo = Prep.tomo
        if Prep is not None and Align is not None:
            self.tomo = Align.tomo

    self.metadata = Import.metadata

    def make_recon_tab(self):

        extend_description_style = {"description_width": "auto"}
        fpath = self.metadata["fpath"]
        fname = self.metadata["fname"]
        self.metadata["opts"] = {}
        self.metadata["methods"] = {}
        self.metadata["save_opts"] = {}

        #tomo_number = int(filter(str.isdigit, box_title))

        radio_recon = RadioButtons(
            options=["Yes", "No"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            value="No",
        )

        radio_recon_fulldataset = RadioButtons(
            options=["Full", "Partial"],
            style=extend_description_style,
            layout=Layout(width="20%"),
            disabled=True,
            value="Full",
        )

        projection_range_x = IntRangeSlider(
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

        projection_range_y = IntRangeSlider(
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

        def activate_box(change):
            if change.new == 0:
                radio_recon_fulldataset.disabled = False
                self.metadata["reconstruct"] = True
                self.metadata["opts"] = {}
                self.metadata["methods"] = {}
                self.metadata["save_opts"] = {}
                save_options_accordion.selected_index = 0
                # self.metadata["methods"]["SIRT_CUDA"] = {}
                options_accordion.selected_index = 0
                methods_accordion.selected_index = 0
            elif change.new == 1:
                radio_recon_fulldataset.disabled = True
                projection_range_x.disabled = True
                projection_range_y.disabled = True
                self.metadata["reconstruct"] = False
                self.metadata.pop("opts")
                self.metadata.pop("methods")
                self.metadata.pop("save_opts")
                save_options_accordion.selected_index = None
                options_accordion.selected_index = None
                methods_accordion.selected_index = None

        def set_projection_ranges(sizeY, sizeX):
            projection_range_x.max = sizeX - 1
            projection_range_y.max = sizeY - 1
            # projection_range_z_recon.max = sizeZ-1
            projection_range_x.value = [0, sizeX - 1]
            projection_range_y.value = [0, sizeY - 1]
            # projection_range_z_recon.value = [0, sizeZ-1]
            self.metadata["prj_range_x"] = projection_range_x.value
            self.metadata["prj_range_y"] = projection_range_y.value

        def load_tif_shape_tag(folder_import=False):
            os.chdir(fpath)
            tiff_count_in_folder = len(glob.glob1(fpath, "*.tif"))
            global sizeY, sizeX
            if folder_import:
                _tomo = td.TomoData(metadata=self.metadata)
                size = _tomo.prj_imgs.shape
                #sizeZ = size[0]
                sizeY = size[1]
                sizeX = size[2]
                set_projection_ranges(sizeY, sizeX)

            else:
                with tf.TiffFile(fname) as tif:
                    if tiff_count_in_folder > 50:
                        sizeX = tif.pages[0].tags["ImageWidth"].value
                        sizeY = tif.pages[0].tags["ImageLength"].value
                        #sizeZ = tiff_count_in_folder # can maybe use this later
                    else:
                        imagesize = tif.pages[0].tags["ImageDescription"]
                        size = json.loads(imagesize.value)["shape"]
                        sizeY = size[1]
                        sizeX = size[2]
                    set_projection_ranges(sizeY, sizeX)

        def load_npy_shape():
            os.chdir(fpath)
            size = np.load(fname, mmap_mode="r").shape
            global sizeY, sizeX
            sizeY = size[1]
            sizeX = size[2]
            set_projection_ranges(sizeY, sizeX)

        def activate_full_partial(change):
            if change.new == 1:
                self.metadata["partial"] = True
                projection_range_x.disabled = False
                projection_range_y.disabled = False
                #projection_range_z_recon.disabled = False
                if fname != "":
                    if fname.__contains__(".tif"):
                        load_tif_shape_tag()
                    elif self.metadata["fname"].__contains__(".npy"):
                        load_npy_shape()
                    else:
                        load_tif_shape_tag(folder_import=True)
            elif change.new == 0:
                self.metadata["partial"] = False
                set_projection_ranges(sizeY, sizeX)
                projection_range_x.disabled = True
                projection_range_y.disabled = True
                #projection_range_z_recon.disabled = True

        self.metadata["partial"] = False
        radio_recon.observe(activate_box, names="index")
        radio_recon_fulldataset.observe(activate_full_partial, names="index")

        #### callbacks for projection range sliders

        @debounce(0.2)
        def projection_range_x_update_dict(change):
            self.metadata["prj_range_x"] = change.new
        projection_range_x.observe(projection_range_x_update_dict, 'value')

        @debounce(0.2)
        def projection_range_y_update_dict(change):
            self.metadata["prj_range_y"] = change.new
        projection_range_y.observe(projection_range_y_update_dict, 'value')

        #### downsampling
        self.metadata["opts"]["downsample"] = False
        self.metadata["opts"]["downsample_factor"] = 1
        def downsample_turn_on(change):
            if change.new == 1:
                self.metadata["opts"]["downsample"] = True
                self.metadata["opts"]["downsample_factor"] = downsample_factor_text.value
                downsample_factor_text.disabled = False
            if change.new == 0:
                self.metadata["opts"]["downsample"] = False
                self.metadata["opts"]["downsample_factor"] = 1
                downsample_factor_text.value = 1
                downsample_factor_text.disabled = True

        downsample_checkbox = Checkbox(description="Downsample?", value=0)
        downsample_checkbox.observe(downsample_turn_on)

        def downsample_factor_update_dict(change):
            self.metadata["opts"]["downsample_factor"] = change.new

        downsample_factor_text = BoundedFloatText(
            value=1, min=0.001, max=1.0, description="Downsampling factor:", disabled=True,
            style=extend_description_style
        )

        downsample_factor_text.observe(downsample_factor_update_dict, names='value')


        #### radio descriptions

        radio_description = (
            "Would you like to reconstruct this dataset?"
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

        #### Saving options

        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_save_dict_on_checkmark(change, opt_list):
            self.metadata["save_opts"] = create_option_dictionary(opt_list)

        save_opts = ["tomo_before", "recon", "tiff", "npy"]
        self.metadata["save_opts"] = {key:None for key in save_opts}

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

        #### Methods checkboxes

        recon_FP_CUDA = Checkbox(description="FP_CUDA")
        recon_BP_CUDA = Checkbox(description="BP_CUDA")
        recon_FBP_CUDA = Checkbox(description="FBP_CUDA")
        ### !!!!!!!! sirt cuda has options - maybe make them into a radio chooser
        recon_SIRT_CUDA = Checkbox(description="SIRT_CUDA")
        recon_SIRT_CUDA_option1 = Checkbox(description="SIRT Plugin-Faster", disabled=False)
        recon_SIRT_CUDA_option2 = Checkbox(description="SIRT 3D-Fastest", disabled=False)
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

        ####### Toggling on options if you select SIRT. Copy the observe function below
        ######## if more options are needed.

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

        [checkbox.observe(create_dict_on_checkmark_no_options) for checkbox in recon_method_list]
        # Makes generator for mapping of options to observe functions.
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

        #### options

        # number of iterations
        self.metadata["opts"]["num_iter"] = 20
        def update_num_iter_dict(change):
            self.metadata["opts"]["num_iter"] = change.new
        number_of_recon_iterations = IntText(
            description="Number of Iterations: ",
            style=extend_description_style,
            value=20,
        )
        number_of_recon_iterations.observe(update_num_iter_dict, names='value')

        #center of rotation
        self.metadata["opts"]["center"] = 0
        def update_center_of_ration_dict(change):
            self.metadata["opts"]["center"] = change.new
        center_of_rotation = IntText(
            description="Center of Rotation: ",
            style=extend_description_style,
            value=self.metadata["opts"]["center"]
        )
        center_of_rotation.observe(update_center_of_ration_dict, names='value')

        self.metadata["opts"]["extra_options"] = None
        def update_extra_options_dict(change):
            self.metadata["opts"]["extra_options"] = change.new
        extra_options = Text(
            description="Extra options: ",
            placeholder='{"MinConstraint": 0}',
            style=extend_description_style,
        )
        extra_options.observe(update_extra_options_dict, names='value')

        downsample_hb = HBox([downsample_checkbox,downsample_factor_text],
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
            [
                projection_range_x,
                projection_range_y,
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

        recon_tab = VBox([recon_initialization_box,options_accordion, methods_accordion,
            save_options_accordion])

        return recon_tab
