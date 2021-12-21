from ipywidgets import *
from . import helpers

extend_description_style = {"description_width": "auto"}

def _init_widgets(obj):
    '''
    Initializes many of the widgets in the Alignment and Recon tabs.
    '''

    # -- Radio to turn on tab ---------------------------------------------
    obj.radio_tab = RadioButtons(
        options=["Yes", "No"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        value="No",
    )
    
    # -- Radio to turn on partial dataset ---------------------------------
    obj.radio_fulldataset = RadioButtons(
        options=["Full", "Partial"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        disabled=True,
        value="Full",
    )

    # -- Radio description, turn on partial dataset ---------------------------
    obj.partial_radio_description = (
        "Would you like to use the full dataset, or a partial dataset?"
    )
    obj.partial_radio_description = HTML(
        value="<style>p{word-wrap: break-word}</style> <p>"
        + obj.partial_radio_description
        + " </p>"
    )

    # -- Plotting --------------------------------------------- 
    obj.plot_prj_images_button = Button(
        description="Click to plot projection images.",
        disabled=False,
        button_style="info",
        tooltip="Plot the prj images to be loaded into alignment.",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )

    obj.plotting_vbox = VBox(
        [
            obj.plot_prj_images_button,
            VBox([]),
            obj.prj_plotter.set_range_button,
        ]
    )

    obj.plotter_accordion = Accordion(
        children=[obj.plotting_vbox],
        selected_index=None,
        layout=Layout(width="100%"),
        titles=("Plot Projection Images",),
    )

    # -- Saving Options -------------------------------------------------------
    obj.save_opts = {key: False for key in obj.save_opts_list}
    obj.save_opts_checkboxes = helpers.create_checkboxes_from_opt_list(
                                                    obj.save_opts_list, 
                                                    obj.save_opts,
                                                    obj)

    # -- Method Options -------------------------------------------------------
    obj.methods_opts = {key: False for key in obj.methods_list}
    obj.methods_checkboxes = helpers.create_checkboxes_from_opt_list(
                                                obj.methods_list, 
                                                obj.methods_opts,
                                                obj)


    # -- Projection Range Sliders --------------------------------------------- 
    # Sliders are defined from the plotter. Probably a better way to go about 
    # this.

    obj.prj_range_x_slider = obj.prj_plotter.prj_range_x_slider
    obj.prj_range_y_slider = obj.prj_plotter.prj_range_y_slider

    # -- Options ---------------------------------------------------------- 

    # Number of iterations
    obj.num_iterations_textbox = IntText(
        description="Number of Iterations: ",
        style=extend_description_style,
        value=obj.num_iter,
    )

    # Center 
    obj.center_of_rotation = FloatText(
        description="Center of Rotation: ",
        style=extend_description_style,
        value=obj.center,
    )

    # Downsampling
    obj.downsample_checkbox = Checkbox(description="Downsample?", value=False)
    obj.downsample_factor_textbox = BoundedFloatText(
        value=obj.downsample_factor,
        min=0.001,
        max=1.0,
        description="Downsample factor:",
        disabled=True,
        style=extend_description_style,
    )

    # Batch size
    obj.batch_size = IntText(
        description="Batch size (for GPU): ",
        style=extend_description_style,
        value=obj.batch_size,
    )

    # X Padding
    obj.paddingX_textbox = IntText(
        description="Padding X (px): ",
        style=extend_description_style,
        value=obj.paddingX,
    )

    # Y Padding
    obj.paddingY_textbox = IntText(
        description="Padding Y (px): ",
        style=extend_description_style,
        value=obj.paddingY,
    )
    # Extra options
    obj.extra_options_textbox = Text(
        description="Extra options: ",
        placeholder='{"MinConstraint": 0}',
        style=extend_description_style,
    )

    # -- Object-specific widgets ---------------------------------------------- 

    if obj.widget_type == "Align":

        # -- Description of turn-on radio -------------------------------------
        obj.radio_description = "Would you like to align this dataset?"
        obj.radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + obj.radio_description
            + " </p>"
        )
        # -- Progress bars and plotting output --------------------------------
        obj.progress_total = IntProgress(description="Recon: ", value=0, min=0, max=1)
        obj.progress_reprj = IntProgress(
            description="Reproj: ", value=0, min=0, max=1
        )
        obj.progress_phase_cross_corr = IntProgress(
            description="Phase Corr: ", value=0, min=0, max=1
        )
        obj.progress_shifting = IntProgress(
            description="Shifting: ", value=0, min=0, max=1
        )
        obj.plot_output1 = Output()
        obj.plot_output2 = Output()

        # -- Button to start alignment ----------------------------------------
        obj.start_button = Button(
            description="After choosing all of the options above, click this button to start the alignment.",
            disabled=True,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start alignment with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        # -- Upsample factor --------------------------------------------------
        obj.upsample_factor_textbox = FloatText(
            description="Upsample Factor: ",
            style=extend_description_style,
            value=obj.upsample_factor,
        )
        
    elif obj.widget_type == "Recon":

        # -- Description of turn-on radio -------------------------------------
        obj.radio_description = "Would you like to reconstruct this dataset?"
        obj.radio_description = HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + obj.radio_description
            + " </p>"
        )
        # -- Button to start reconstruction -----------------------------------
        obj.start_button = Button(
            description="After choosing all of the options above, click this button to start the reconstruction.",
            disabled=True,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start reconstruction with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )