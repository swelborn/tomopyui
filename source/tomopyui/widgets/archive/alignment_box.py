from ipywidgets import *
import functools
import tomopy.data.tomodata as td
from tomopy.data.tomoalign import TomoAlign, init_new_from_prior

# import tomopy.data.tomo as td

# TODO: This is very disorganized. Try to bring some order/organization.


def alignment_box(meta, aligned_tomo_list, tomo, widget_linker, generalmetadata):
    main_logger = generalmetadata["main_logger"]
    main_handler = generalmetadata["main_handler"]

    projection_range_x_movie = widget_linker["projection_range_x_movie"]
    projection_range_y_movie = widget_linker["projection_range_y_movie"]
    projection_range_theta_movie = widget_linker["projection_range_theta_movie"]
    skip_theta_movie = widget_linker["skip_theta_movie"]

    extend_description_style = {"description_width": "auto"}

    radio_align = RadioButtons(
        options=["Yes", "No"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        value="No",
    )
    radio_align_fulldataset = RadioButtons(
        options=["Full", "Partial"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        disabled=True,
        value="Full",
    )
    # make sure the file choosers are going to the correct directory.
    projection_range_x_alignment = IntRangeSlider(
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

    projection_range_y_alignment = IntRangeSlider(
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

    number_of_align_iterations = IntText(
        description="Number of Iterations: ",
        style=extend_description_style,
        value=20,
    )
    center_of_rotation = IntText(
        description="Center of Rotation: ",
        style=extend_description_style,
        value=tomo.prj_imgs.shape[2] / 2,
    )
    upsample_factor = IntText(
        description="Upsample Factor: ", style=extend_description_style, value=1
    )
    batch_size = IntText(
        description="Batch Size (!: if too small, can crash memory) ",
        style=extend_description_style,
        value=20,
        layout=Layout(width="auto"),
    )
    paddingX = IntText(
        description="Padding X: ", style=extend_description_style, value=10
    )
    paddingY = IntText(
        description="Padding Y: ", style=extend_description_style, value=10
    )
    extra_options = Text(
        description="Extra options: ",
        placeholder='{"MinConstraint": 0}',
        style=extend_description_style,
    )

    align_start_button = Button(
        description="After choosing all of the options above, click this button to start the alignment.",
        disabled=True,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Make sure to choose all of the buttons above before clicking this button",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )

    # enable the alignment gui if on.
    def radio_align_true(change):
        if change.new == 0:
            radio_align_fulldataset.disabled = False
            align_start_button.disabled = False
            meta["aligndata"] = True
            meta["methods"]["SIRT_CUDA"] = {}
            other_options_accordion.selected_index = 0
            grid_methods_accordion.selected_index = 0
            save_options_accordion.selected_index = 0

        elif change.new == 1:
            radio_align_fulldataset.disabled = True
            align_start_button.disabled = True
            projection_range_x_alignment.disabled = True
            projection_range_y_alignment.disabled = True
            load_range_from_above.disabled = True
            meta["aligndata"] = False
            other_options_accordion.selected_index = None
            grid_methods_accordion.selected_index = None
            save_options_accordion.selected_index = None

    def radio_align_full_partial(change):
        if change.new == 1:
            projection_range_x_alignment.disabled = False
            projection_range_y_alignment.disabled = False
            load_range_from_above.disabled = False
            meta["aligndata"] = True
            load_range_from_above.description = (
                "Click to load projection range from above."
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
            projection_range_x_alignment.value = [0, tomo.prj_imgs.shape[2] - 1]
            projection_range_x_alignment.disabled = True
            projection_range_y_alignment.value = [0, tomo.prj_imgs.shape[1] - 1]
            projection_range_y_alignment.disabled = True
            load_range_from_above.disabled = True
            meta["aligndata"] = False

    def load_range_from_above_onclick(self):
        if self.button_style == "info":
            global range_y_link, range_x_link
            range_y_link = link(
                (projection_range_y_movie, "value"),
                (projection_range_y_alignment, "value"),
            )
            range_x_link = link(
                (projection_range_x_movie, "value"),
                (projection_range_x_alignment, "value"),
            )
            self.button_style = "success"
            self.description = "Linked ranges. Click again to unlink."
            self.icon = "link"
        elif self.button_style == "success":
            range_y_link.unlink()
            range_x_link.unlink()
            projection_range_x_alignment.value = [0, tomo.prj_imgs.shape[2] - 1]
            projection_range_y_alignment.value = [0, tomo.prj_imgs.shape[1] - 1]
            self.button_style = "info"
            self.description = "Unlinked ranges. Click again to link."
            self.icon = "unlink"

    method_output = Output()
    output0 = Output()
    output1 = Output()
    output2 = Output()

    #################### START ALIGNMENT ######################################
    def set_options_and_run_align(self):
        self.icon = "fas fa-cog fa-spin fa-lg"
        self.description = "Setting options and loading data into alignment algorithm."
        meta["opts"]["num_iter"] = number_of_align_iterations.value
        meta["opts"]["center"] = center_of_rotation.value
        meta["opts"]["prj_range_x"] = projection_range_x_alignment.value
        meta["opts"]["prj_range_y"] = projection_range_y_alignment.value
        meta["opts"]["upsample_factor"] = upsample_factor.value
        meta["opts"]["pad"] = (
            paddingX.value,
            paddingY.value,
        )
        meta["opts"]["batch_size"] = batch_size.value
        meta["opts"]["extra_options"] = extra_options.value
        meta["callbacks"]["button"] = self
        meta["callbacks"]["methodoutput"] = method_output
        meta["callbacks"]["output0"] = output0
        meta["callbacks"]["output1"] = output1
        meta["callbacks"]["output2"] = output2
        meta["opts"]["downsample"] = downsample_checkbox.value
        meta["opts"]["downsample_factor"] = downsample_factor_text.value
        if len(meta["methods"]) > 1:
            meta["alignmultiple"] = True
        try:
            self.description = "Aligning your data."
            align_number = meta["align_number"]
            if align_number == 0:
                aligned_tomo_list.append(TomoAlign(tomo, meta))
            else:
                aligned_tomo_list.append(
                    init_new_from_prior(aligned_tomo_list[align_number - 1], meta)
                )
            self.button_style = "success"
            self.icon = "fa-check-square"
            self.description = "Finished alignment. Click to run second alignment with updated parameters."
            meta["align_number"] += 1
        except:
            with output0:
                self.button_style = "warning"
                self.icon = "exclamation-triangle"
                self.description = "Something went wrong."

    ############################# METHOD CHOOSER GRID ############################
    grid_alignment = GridspecLayout(2, 3)
    # align_FBP_CUDA = Checkbox(description="FBP_CUDA")
    # align_FBP_CUDA_option1 = Checkbox(description="option1", disabled=True)
    # align_FBP_CUDA_option2 = Checkbox(description="option2", disabled=True)
    # align_FBP_CUDA_option3 = Checkbox(description="option3", disabled=True)
    # align_FBP_CUDA_option_list = [
    #     align_FBP_CUDA_option1,
    #     align_FBP_CUDA_option2,
    #     align_FBP_CUDA_option3,
    # ]

    align_SIRT_CUDA = Checkbox(description="SIRT_CUDA", value=1)
    align_SIRT_CUDA_option1 = Checkbox(description="Faster", disabled=False)
    align_SIRT_CUDA_option2 = Checkbox(description="Fastest", disabled=False)
    align_SIRT_CUDA_option3 = Checkbox(description="option3", disabled=False)
    align_SIRT_CUDA_option_list = [align_SIRT_CUDA_option1, align_SIRT_CUDA_option2]

    # align_SART_CUDA = Checkbox(description="SART_CUDA")
    # align_SART_CUDA_option1 = Checkbox(description="option1", disabled=True)
    # align_SART_CUDA_option2 = Checkbox(description="option2", disabled=True)
    # align_SART_CUDA_option3 = Checkbox(description="option3", disabled=True)
    # align_SART_CUDA_option_list = [
    #     align_SART_CUDA_option1,
    #     align_SART_CUDA_option2,
    #     align_SART_CUDA_option3,
    # ]

    # align_CGLS_CUDA = Checkbox(description="CGLS_CUDA")
    # align_CGLS_CUDA_option1 = Checkbox(description="option1", disabled=True)
    # align_CGLS_CUDA_option2 = Checkbox(description="option2", disabled=True)
    # align_CGLS_CUDA_option3 = Checkbox(description="option3", disabled=True)
    # align_CGLS_CUDA_option_list = [
    #     align_CGLS_CUDA_option1,
    #     align_CGLS_CUDA_option2,
    #     align_CGLS_CUDA_option3,
    # ]

    # align_MLEM_CUDA = Checkbox(description="MLEM_CUDA")
    # align_MLEM_CUDA_option1 = Checkbox(description="option1", disabled=True)
    # align_MLEM_CUDA_option2 = Checkbox(description="option2", disabled=True)
    # align_MLEM_CUDA_option3 = Checkbox(description="option3", disabled=True)
    # align_MLEM_CUDA_option_list = [
    #     align_MLEM_CUDA_option1,
    #     align_MLEM_CUDA_option2,
    #     align_MLEM_CUDA_option3,
    # ]

    align_method_list = [
        # align_FBP_CUDA,
        align_SIRT_CUDA,
        # align_SART_CUDA,
        # align_CGLS_CUDA,
        # align_MLEM_CUDA,
    ]

    def toggle_on(change, opt_list, dictname):
        if change.new == 1:
            meta["methods"][dictname] = {}
            for option in opt_list:
                option.disabled = False
        if change.new == 0:
            meta["methods"].pop(dictname)
            for option in opt_list:
                option.value = 0
                option.disabled = True

    # align_FBP_CUDA.observe(
    #     functools.partial(
    #         toggle_on, opt_list=align_FBP_CUDA_option_list, dictname="FBP_CUDA"
    #     ),
    #     names=["value"],
    # )
    align_SIRT_CUDA.observe(
        functools.partial(
            toggle_on, opt_list=align_SIRT_CUDA_option_list, dictname="SIRT_CUDA"
        ),
        names=["value"],
    )
    # align_SART_CUDA.observe(
    #     functools.partial(
    #         toggle_on, opt_list=align_SART_CUDA_option_list, dictname="SART_CUDA"
    #     ),
    #     names=["value"],
    # )
    # align_CGLS_CUDA.observe(
    #     functools.partial(
    #         toggle_on, opt_list=align_CGLS_CUDA_option_list, dictname="CGLS_CUDA"
    #     ),
    #     names=["value"],
    # )
    # align_MLEM_CUDA.observe(
    #     functools.partial(
    #         toggle_on, opt_list=align_MLEM_CUDA_option_list, dictname="MLEM_CUDA"
    #     ),
    #     names=["value"],
    # )

    def create_option_dictionary(opt_list):
        opt_dictionary = {opt.description: opt.value for opt in opt_list}
        return opt_dictionary

    def create_dict_on_checkmark(change, opt_list, dictname):
        meta["methods"][dictname] = create_option_dictionary(opt_list)

    # Makes generator for mapping of options to observe functions.

    # list(
    #     (
    #         opt.observe(
    #             functools.partial(
    #                 create_dict_on_checkmark,
    #                 opt_list=align_FBP_CUDA_option_list,
    #                 dictname="FBP_CUDA",
    #             ),
    #             names=["value"],
    #         )
    #         for opt in align_FBP_CUDA_option_list
    #     )
    # )
    list(
        (
            opt.observe(
                functools.partial(
                    create_dict_on_checkmark,
                    opt_list=align_SIRT_CUDA_option_list,
                    dictname="SIRT_CUDA",
                ),
                names=["value"],
            )
            for opt in align_SIRT_CUDA_option_list
        )
    )
    # list(
    #     (
    #         opt.observe(
    #             functools.partial(
    #                 create_dict_on_checkmark,
    #                 opt_list=align_SART_CUDA_option_list,
    #                 dictname="SART_CUDA",
    #             ),
    #             names=["value"],
    #         )
    #         for opt in align_SART_CUDA_option_list
    #     )
    # )
    # list(
    #     (
    #         opt.observe(
    #             functools.partial(
    #                 create_dict_on_checkmark,
    #                 opt_list=align_CGLS_CUDA_option_list,
    #                 dictname="CGLS_CUDA",
    #             ),
    #             names=["value"],
    #         )
    #         for opt in align_CGLS_CUDA_option_list
    #     )
    # )
    # list(
    #     (
    #         opt.observe(
    #             functools.partial(
    #                 create_dict_on_checkmark,
    #                 opt_list=align_MLEM_CUDA_option_list,
    #                 dictname="MLEM_CUDA",
    #             ),
    #             names=["value"],
    #         )
    #         for opt in align_MLEM_CUDA_option_list
    #     )
    # )

    def fill_grid(method, opt_list, linenumber, grid):
        grid[linenumber, 0] = method
        i = 1
        for option in opt_list:
            grid[linenumber, i] = option
            i += 1

    # fill_grid(align_FBP_CUDA, align_FBP_CUDA_option_list, 1, grid_alignment)
    fill_grid(align_SIRT_CUDA, align_SIRT_CUDA_option_list, 1, grid_alignment)
    # fill_grid(align_SART_CUDA, align_SART_CUDA_option_list, 3, grid_alignment)
    # fill_grid(align_CGLS_CUDA, align_CGLS_CUDA_option_list, 4, grid_alignment)
    # fill_grid(align_MLEM_CUDA, align_MLEM_CUDA_option_list, 5, grid_alignment)

    grid_column_headers = ["Method", "Option 1", "Option 2"]
    for i, method in enumerate(grid_column_headers):
        grid_alignment[0, i] = Label(
            value=grid_column_headers[i], layout=Layout(justify_content="center")
        )
    ##############################Alignment start button???#######################
    radio_align.observe(radio_align_true, names="index")
    radio_align_fulldataset.observe(radio_align_full_partial, names="index")
    load_range_from_above.on_click(load_range_from_above_onclick)
    align_start_button.on_click(set_options_and_run_align)

    #######################DOWNSAMPLE CHECKBOX############################
    def downsample_turn_on(change):
        if change.new == 1:
            meta["opts"]["downsample"] = True
            downsample_factor_text.disabled = False
        if change.new == 0:
            meta["opts"]["downsample"] = False
            downsample_factor_text.disabled = True

    downsample_checkbox = Checkbox(description="Downsample?", value=0)
    meta["opts"]["downsample"] = False
    downsample_checkbox.observe(downsample_turn_on)

    downsample_factor_text = BoundedFloatText(
        value=0.5, min=0.001, max=1.0, description="Downsampling factor:", disabled=True,
        style=extend_description_style
    )

    ######################SAVING OPTIONS########################

    def create_option_dictionary(opt_list):
        opt_dictionary = {opt.description: opt.value for opt in opt_list}
        return opt_dictionary

    def create_save_dict_on_checkmark(change, opt_list):
        meta["save_opts"] = create_option_dictionary(opt_list)

    save_opts = ["tomo_after", "tomo_before", "recon", "tiff", "npy"]

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
        titles=("Save options",),
    )

    ####################    ALIGNMENT BOX ORGANIZATION   ########################
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
            HBox(
                [load_range_from_above],
                justify_content="center",
                align_content="center",
            ),
            projection_range_x_alignment,
            projection_range_y_alignment,
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
    grid_methods_accordion = Accordion(
        children=[grid_alignment], selected_index=None, titles=("Alignment Methods",)
    )

    other_options_accordion = Accordion(
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

    box = VBox(
        children=[
            hb1,
            grid_methods_accordion,
            save_options_accordion,
            other_options_accordion,
            hb2,
            method_output,
            output1,
            output2,
        ]
    )

    return box
