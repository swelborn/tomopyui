from ipywidgets import *
from .._shared import helpers
def _init_widgets(Align):
    '''
    Initializes many of the widgets in the Alignment tab.
    '''
    extend_description_style = {"description_width": "auto"}
    Align.save_opts = {key: False for key in Align.save_opts_list}
    Align.save_opts_checkboxes = helpers.create_checkboxes_from_opt_list(
                                                    Align.save_opts_list, 
                                                    Align.save_opts,
                                                    Align)
    Align.methods_opts = {key: False for key in Align.methods_list}
    Align.methods_checkboxes = helpers.create_checkboxes_from_opt_list(
                                                Align.methods_list, 
                                                Align.methods_opts,
                                                Align)
    Align.progress_total = IntProgress(description="Recon: ", value=0, min=0, max=1)

    # Sliders are defined from the plotter. Probably a better way to go about 
    # this. I could not find a way to link both their

    Align.prj_range_x_slider = Align.prj_plotter.prj_range_x_slider
    Align.prj_range_y_slider = Align.prj_plotter.prj_range_y_slider


    # Align.prj_range_x_slider = IntRangeSlider(
    #     value=[0, 10],
    #     min=0,
    #     max=10,
    #     step=1,
    #     description="Projection X Range:",
    #     disabled=True,
    #     continuous_update=False,
    #     orientation="horizontal",
    #     readout=True,
    #     readout_format="d",
    #     layout=Layout(width="100%"),
    #     style=extend_description_style,
    # )
    # Align.prj_range_y_slider = IntRangeSlider(
    #     value=[0, 10],
    #     min=0,
    #     max=10,
    #     step=1,
    #     description="Projection Y Range:",
    #     disabled=True,
    #     continuous_update=False,
    #     orientation="horizontal",
    #     readout=True,
    #     readout_format="d",
    #     layout=Layout(width="100%"),
    #     style=extend_description_style,
    # )

    Align.progress_reprj = IntProgress(
        description="Reproj: ", value=0, min=0, max=1
    )
    Align.progress_phase_cross_corr = IntProgress(
        description="Phase Corr: ", value=0, min=0, max=1
    )
    Align.progress_shifting = IntProgress(
        description="Shifting: ", value=0, min=0, max=1
    )
    Align.plot_output1 = Output()
    Align.plot_output2 = Output()

    Align.plot_prj_images_button = Button(
        description="Click to plot projection images.",
        disabled=False,
        button_style="info",
        tooltip="Plot the prj images to be loaded into alignment.",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )

    Align.plotting_vbox = VBox(
        [
            Align.plot_prj_images_button,
            VBox([]),
            Align.prj_plotter.set_range_button,
        ]
    )

    Align.plotter_accordion = Accordion(
        children=[Align.plotting_vbox],
        selected_index=None,
        layout=Layout(width="100%"),
        titles=("Plot Projection Images",),
    )

    Align.radio_align_fulldataset = RadioButtons(
        options=["Full", "Partial"],
        style=extend_description_style,
        layout=Layout(width="20%"),
        disabled=True,
        value="Full",
    )

    Align.align_start_button = Button(
        description="After choosing all of the options above, click this button to start the alignment.",
        disabled=True,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Start aligning things with this button.",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )