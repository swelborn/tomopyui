from ipywidgets import *
import functools
import tomopy.data.tomodata as td
from .plot_aligned_data import plot_aligned_data

# TODO: This is very disorganized. Try to bring some order/organization.


def recon_box(
    reconmeta,
    generalmetadata,
    importmetadata,
    alignmentmetadata,
    alignmentdata,
    aligned_tomo_list,
    widget_linker,
):
    plot_vbox, recon_files = plot_aligned_data(
        reconmeta,
        alignmentmetadata,
        importmetadata,
        generalmetadata,
        alignmentdata,
        widget_linker,
    )

    main_logger = generalmetadata["main_logger"]
    main_handler = generalmetadata["main_handler"]

    # projection_range_x_movie = widget_linker["projection_range_x_movie"]
    # projection_range_y_movie = widget_linker["projection_range_y_movie"]
    # projection_range_theta_movie = widget_linker["projection_range_theta_movie"]
    # skip_theta_movie = widget_linker["skip_theta_movie"]

    extend_description_style = {"description_width": "auto"}

    ############ Perform recon? Y/N button
    radio_recon = RadioButtons(
        options=["Yes", "No"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        value="No",
    )
    ############ Full dataset? Full/partial radio
    radio_recon_fulldataset = RadioButtons(
        options=["Full", "Partial"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        disabled=True,
        value="Full",
    )

    ############ If partial, use sliders here
    projection_range_x_recon = IntRangeSlider(
        value=[0, tomo.prj_imgs.shape[2] - 1],
        min=0,
        max=tomo.prj_imgs.shape[2] - 1,
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
        value=[0, tomo.prj_imgs.shape[1] - 1],
        min=0,
        max=tomo.prj_imgs.shape[1] - 1,
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

    load_range_from_above = Button(
        description="Click to load projection range from above.",
        disabled=True,
        button_style="info",
        tooltip="Make sure to choose all of the buttons above before clicking this button",
        icon="",
        layout=Layout(width="95%", justify_content="center"),
    )

    number_of_recon_iterations = IntText(
        description="Number of Iterations: ",
        style=extend_description_style,
        value=20,
    )
    center_of_rotation = IntText(
        description="Center of Rotation: ",
        style=extend_description_style,
        value=tomo.prj_imgs.shape[2] / 2,
    )

    extra_options = Text(
        description="Extra options: ",
        placeholder='{"MinConstraint": 0}',
        style=extend_description_style,
    )

    recon_start_button = Button(
        description="After choosing all of the options above, click this button to start the reconstruction.",
        disabled=True,
        button_style="info",
        tooltip="Make sure to choose all of the buttons above before clicking this button",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )

    # enable the alignment gui if on.
    def radio_recon_true(change):
        if change.new == 0:
            radio_recon_fulldataset.disabled = False
            recon_start_button.disabled = False
            reconmeta["recondata"] = True
            reconmeta["methods"]["SIRT_CUDA"] = {}
            other_options_accordion.selected_index = 0
            methods_accordion.selected_index = 0
            save_options_accordion.selected_index = 0

        elif change.new == 1:
            radio_recon_fulldataset.disabled = True
            recon_start_button.disabled = True
            projection_range_x_recon.disabled = True
            projection_range_y_recon.disabled = True
            load_range_from_above.disabled = True
            reconmeta["recondata"] = False
            other_options_accordion.selected_index = None
            methods_accordion.selected_index = None
            save_options_accordion.selected_index = None

    ####!!!!!!!!!!!!!! Fix plotting widget link
    def radio_recon_full_partial(change):
        if change.new == 1:
            projection_range_x_recon.disabled = False
            projection_range_y_recon.disabled = False
            load_range_from_above.disabled = False
            reconmeta["recondata"] = True
            load_range_from_above.description = (
                "Click to load projection range from plotting screen."
            )
            load_range_from_above.icon = ""
        elif change.new == 0:
            if "range_y_link" in locals() or "range_y_link" in globals():
                range_y_link.unlink()
                range_x_link.unlink()
                load_range_from_above.button_style = "info"
                load_range_from_above.description = (
                    "Unlinked ranges. Enable partial range to link again."
                )
                load_range_from_above.icon = "unlink"
            projection_range_x_recon.value = [0, tomo.prj_imgs.shape[2] - 1]
            projection_range_x_recon.disabled = True
            projection_range_y_recon.value = [0, tomo.prj_imgs.shape[1] - 1]
            projection_range_y_recon.disabled = True
            load_range_from_above.disabled = True
            meta["aligndata"] = False

    def load_range_from_above_onclick(self):
        if self.button_style == "info":
            global range_y_link, range_x_link
            range_y_link = link(
                (projection_range_y_movie, "value"),
                (projection_range_y_recon, "value"),
            )
            range_x_link = link(
                (projection_range_x_movie, "value"),
                (projection_range_x_recon, "value"),
            )
            self.button_style = "success"
            self.description = "Linked ranges. Click again to unlink."
            self.icon = "link"
        elif self.button_style == "success":
            range_y_link.unlink()
            range_x_link.unlink()
            projection_range_x_recon.value = [0, tomo.prj_imgs.shape[2] - 1]
            projection_range_y_recon.value = [0, tomo.prj_imgs.shape[1] - 1]
            self.button_style = "info"
            self.description = "Unlinked ranges. Click again to link."
            self.icon = "unlink"

    method_output = Output()
    output0 = Output()

    #################### START Recon #######################
    def set_options_and_run_align(self):
        self.icon = "fas fa-cog fa-spin fa-lg"
        self.description = (
            "Setting options and loading data into memory for reconstruction."
        )
        reconmeta["opts"]["num_iter"] = number_of_recon_iterations.value
        reconmeta["opts"]["center"] = center_of_rotation.value
        reconmeta["opts"]["prj_range_x"] = projection_range_x_recon.value
        reconmeta["opts"]["prj_range_y"] = projection_range_y_recon.value
        reconmeta["opts"]["extra_options"] = extra_options.value
        #!!!!!!!!! what do these call backs do in recon.
        reconmeta["callbacks"]["button"] = self
        reconmeta["callbacks"]["methodoutput"] = method_output
        reconmeta["callbacks"]["output0"] = output0
        reconmeta["opts"]["downsample"] = downsample_checkbox.value
        reconmeta["opts"]["downsample_factor"] = downsample_factor_text.value
        if len(reconmeta["methods"]) > 1:
            reconmeta["reconmultiple"] = True
        try:
            self.description = "Reconstructing your data."
            ##### reconstruction function goes here.
            # aligned_tomo_list.append(TomoAlign(tomo, reconmeta))
            self.button_style = "success"
            self.icon = "fa-check-square"
            self.description = "Finished reconstruction."
        except:
            self.button_style = "warning"
            self.icon = "exclamation-triangle"
            self.description = "Something went wrong."

    ############################# METHOD CHOOSER BOX ############################
    recon_FP_CUDA = Checkbox(description="FP_CUDA")
    recon_BP_CUDA = Checkbox(description="BP_CUDA")
    recon_FBP_CUDA = Checkbox(description="FBP_CUDA")
    ### !!!!!!!! sirt cuda has options - maybe make them into a radio chooser
    recon_SIRT_CUDA = Checkbox(description="SIRT_CUDA")
    recon_SIRT_CUDA_option1 = Checkbox(description="SIRT Plugin-Faster", disabled=False)
    recon_SIRT_CUDA_option2 = Checkbox(description="SIRT 3D-Fastest", disabled=False)
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
            reconmeta["methods"][dictname] = {}
            for option in opt_list:
                option.disabled = False
        if change.new == 0:
            reconmeta["methods"].pop(dictname)
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
        reconmeta["methods"][dictname] = create_option_dictionary(opt_list)

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

    sirt_hbox = HBox([recon_SIRT_CUDA])

    recon_method_box = VBox(
        [
            VBox(recon_method_list, layout=widgets.Layout(flex_flow="row wrap")),
            sirt_hbox,
        ]
    )

    ##############################Alignment start button???#######################
    radio_align.observe(radio_recon_true, names="index")
    radio_recon_fulldataset.observe(radio_recon_full_partial, names="index")
    load_range_from_above.on_click(load_range_from_above_onclick)
    recon_start_button.on_click(set_options_and_run_align)

    #######################DOWNSAMPLE CHECKBOX############################
    def downsample_turn_on(change):
        if change.new == 1:
            reconmeta["opts"]["downsample"] = True
            downsample_factor_text.disabled = False
        if change.new == 0:
            reconmeta["opts"]["downsample"] = False
            downsample_factor_text.disabled = True

    downsample_checkbox = Checkbox(description="Downsample?", value=0)
    reconmeta["opts"]["downsample"] = False
    downsample_checkbox.observe(downsample_turn_on)

    downsample_factor_text = BoundedFloatText(
        value=0.5, min=0.001, max=1.0, description="Downsampling factor:", disabled=True
    )

    ######################SAVING OPTIONS########################

    def create_option_dictionary(opt_list):
        opt_dictionary = {opt.description: opt.value for opt in opt_list}
        return opt_dictionary

    def create_save_dict_on_checkmark(change, opt_list):
        reconmeta["save_opts"] = create_option_dictionary(opt_list)

    save_opts = ["tomo_before", "recon", "tiff", "npy"]

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

    ####################    ALIGNMENT BOX ORGANIZATION   ########################
    radio_description = "Reconstruct your data?"
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
            HBox(
                [load_range_from_above],
                justify_content="center",
                align_content="center",
            ),
            projection_range_x_recon,
            projection_range_y_recon,
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
            radio_recon_fulldataset,
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
        [recon_start_button], layout=Layout(width="auto", justify_content="center")
    )
    methods_accordion = Accordion(
        children=[recon_method_box], selected_index=None, titles=("Methods",)
    )

    other_options_accordion = Accordion(
        children=[
            VBox(
                [
                    HBox(
                        [
                            number_of_recon_iterations,
                            center_of_rotation,
                            upsample_factor,
                        ],
                        layout=Layout(
                            flex_wrap="wrap", justify_content="space-between"
                        ),
                    ),
                    HBox(
                        [
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
        titles=("Other options",),
    )

    recon_box = VBox(
        children=[
            hb1,
            methods_accordion,
            save_options_accordion,
            other_options_accordion,
            hb2,
            method_output,
        ]
    )

    return recon_box, plot_vbox, recon_files
