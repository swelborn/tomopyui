import tifffile as tf
from ipywidgets import *
import glob
from .debouncer import debounce
import functools
import json


def generate_alignment_box(recon_tomo_metadata):

    extend_description_style = {"description_width": "auto"}
    fpath = recon_tomo_metadata["fpath"]
    fname = recon_tomo_metadata["fname"]
    recon_tomo_metadata["opts"] = {}
    recon_tomo_metadata["methods"] = {}
    recon_tomo_metadata["save_opts"] = {}

    #tomo_number = int(filter(str.isdigit, box_title))

    radio_alignment = RadioButtons(
        options=["Yes", "No"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        value="No",
    )

    radio_alignment_fulldataset = RadioButtons(
        options=["Full", "Partial"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        disabled=True,
        value="Full",
    )

    projection_range_x_recon = IntRangeSlider(
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

    projection_range_y_recon = IntRangeSlider(
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
            radio_alignment_fulldataset.disabled = False
            recon_tomo_metadata["reconstruct"] = True
            recon_tomo_metadata["opts"] = {}
            recon_tomo_metadata["methods"] = {}
            recon_tomo_metadata["save_opts"] = {}
            save_options_accordion.selected_index = 0
            # recon_tomo_metadata["methods"]["SIRT_CUDA"] = {}
            options_accordion.selected_index = 0
            methods_accordion.selected_index = 0
        elif change.new == 1:
            radio_alignment_fulldataset.disabled = True
            projection_range_x_recon.disabled = True
            projection_range_y_recon.disabled = True
            recon_tomo_metadata["reconstruct"] = False
            recon_tomo_metadata.pop("opts")
            recon_tomo_metadata.pop("methods")
            recon_tomo_metadata.pop("save_opts")
            save_options_accordion.selected_index = None
            options_accordion.selected_index = None
            methods_accordion.selected_index = None

    def set_projection_ranges(sizeY, sizeX):
        projection_range_x_recon.max = sizeX - 1
        projection_range_y_recon.max = sizeY - 1
        # projection_range_z_recon.max = sizeZ-1
        projection_range_x_recon.value = [0, sizeX - 1]
        projection_range_y_recon.value = [0, sizeY - 1]
        # projection_range_z_recon.value = [0, sizeZ-1]
        recon_tomo_metadata["prj_range_x"] = projection_range_x_recon.value
        recon_tomo_metadata["prj_range_y"] = projection_range_y_recon.value

    def load_tif_shape_tag(folder_import=False):
        os.chdir(fpath)
        tiff_count_in_folder = len(glob.glob1(fpath, "*.tif"))
        global sizeY, sizeX
        if folder_import:
            _tomo = td.TomoData(metadata=recon_tomo_metadata)
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
            recon_tomo_metadata["partial"] = True
            projection_range_x_recon.disabled = False
            projection_range_y_recon.disabled = False
            #projection_range_z_recon.disabled = False
            if fname != "":
                if fname.__contains__(".tif"):
                    load_tif_shape_tag()
                elif recon_tomo_metadata["fname"].__contains__(".npy"):
                    load_npy_shape()
                else:
                    load_tif_shape_tag(folder_import=True)
        elif change.new == 0:
            recon_tomo_metadata["partial"] = False
            set_projection_ranges(sizeY, sizeX)
            projection_range_x_recon.disabled = True
            projection_range_y_recon.disabled = True
            #projection_range_z_recon.disabled = True

    recon_tomo_metadata["partial"] = False
    radio_alignment.observe(activate_box, names="index")
    radio_alignment_fulldataset.observe(activate_full_partial, names="index")

    #### callbacks for projection range sliders

    @debounce(0.2)
    def projection_range_x_update_dict(change):
        recon_tomo_metadata["prj_range_x"] = change.new
    projection_range_x_recon.observe(projection_range_x_update_dict, 'value')

    @debounce(0.2)
    def projection_range_y_update_dict(change):
        recon_tomo_metadata["prj_range_y"] = change.new
    projection_range_y_recon.observe(projection_range_y_update_dict, 'value')

    #### downsampling
    recon_tomo_metadata["opts"]["downsample"] = False
    recon_tomo_metadata["opts"]["downsample_factor"] = 1
    def downsample_turn_on(change):
        if change.new == 1:
            recon_tomo_metadata["opts"]["downsample"] = True
            recon_tomo_metadata["opts"]["downsample_factor"] = downsample_factor_text.value
            downsample_factor_text.disabled = False
        if change.new == 0:
            recon_tomo_metadata["opts"]["downsample"] = False
            recon_tomo_metadata["opts"]["downsample_factor"] = 1
            downsample_factor_text.value = 1
            downsample_factor_text.disabled = True

    downsample_checkbox = Checkbox(description="Downsample?", value=0)
    downsample_checkbox.observe(downsample_turn_on)

    def downsample_factor_update_dict(change):
        recon_tomo_metadata["opts"]["downsample_factor"] = change.new

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
        recon_tomo_metadata["save_opts"] = create_option_dictionary(opt_list)

    save_opts = ["tomo_before", "recon", "tiff", "npy"]
    recon_tomo_metadata["save_opts"] = {key:None for key in save_opts}

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
            recon_tomo_metadata["methods"][dictname] = {}
            for option in opt_list:
                option.disabled = False
        if change.new == 0:
            recon_tomo_metadata["methods"].pop(dictname)
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
        recon_tomo_metadata["methods"][dictname] = create_option_dictionary(opt_list)

    def create_dict_on_checkmark_no_options(change):
        if change.new == True:
            recon_tomo_metadata["methods"][change.owner.description] = {}
        if change.new == False:
            recon_tomo_metadata["methods"].pop(change.owner.description)

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
    recon_tomo_metadata["opts"]["num_iter"] = 20
    def update_num_iter_dict(change):
        recon_tomo_metadata["opts"]["num_iter"] = change.new
    number_of_recon_iterations = IntText(
        description="Number of Iterations: ",
        style=extend_description_style,
        value=20,
    )
    number_of_recon_iterations.observe(update_num_iter_dict, names='value')

    #center of rotation
    recon_tomo_metadata["opts"]["center"] = 0
    def update_center_of_ration_dict(change):
        recon_tomo_metadata["opts"]["center"] = change.new
    center_of_rotation = IntText(
        description="Center of Rotation: ",
        style=extend_description_style,
        value=recon_tomo_metadata["opts"]["center"]
    )
    center_of_rotation.observe(update_center_of_ration_dict, names='value')

    recon_tomo_metadata["opts"]["extra_options"] = None
    def update_extra_options_dict(change):
        recon_tomo_metadata["opts"]["extra_options"] = change.new
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
            projection_range_x_recon,
            projection_range_y_recon,
        ],
        layout=Layout(width="30%"),
        justify_content="center",
        align_items="space-between",
    )

    recon_initialization_box = HBox(
        [
            radio_description,
            radio_alignment,
            partial_radio_description,
            radio_alignment_fulldataset,
            sliders_box,
        ],
        layout=Layout(
            width="auto",
            justify_content="center",
            align_items="center",
            flex="flex-grow",
        ),
    )

    recon_box = VBox([recon_initialization_box,options_accordion, methods_accordion,
        save_options_accordion])




    return recon_box
