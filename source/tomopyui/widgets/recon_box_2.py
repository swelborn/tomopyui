from ipywidgets import *
import functools
import tomopy.data.tomodata as td
from .plot_aligned_data import plot_aligned_data
from .multiple_recon import generate_recon_box
# TODO: This is very disorganized. Try to bring some order/organization.


def recon_box(
    reconmetadata,
    generalmetadata,
    importmetadata,
    alignmentmetadata,
    alignmentdata,
    widget_linker,
):
    plot_vbox, recon_files = plot_aligned_data(
        reconmetadata,
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
    recon_box = []
    recon_box_titles = []

    def make_recon_boxes(self):
        self.icon = "fas fa-cog fa-spin fa-lg"
        self.button_style = "info"
        recon_box = []
        recon_box_titles = []
        for key in reconmetadata["tomo"]:
            if "fpath" in reconmetadata["tomo"][key]:
                recon_box.append(generate_recon_box(reconmetadata["tomo"][key]))
                if "fname" in reconmetadata["tomo"][key]:
                    recon_box_titles.append(reconmetadata["tomo"][key]["fname"])
                else:
                    recon_box_titles.append(reconmetadata["tomo"][key]["fpath"])
        recon_tab.children = recon_box
        recon_tab.titles = recon_box_titles
        self.icon = "fa-check-square"
        self.button_style = "success"
        self.description = "Make your edits to the reconstruction options below. Click this again to upload more data."
        recon_tab_vbox.children = recon_tab_vbox.children + (start_recon_button,)
    make_recon_boxes_button = Button(
        description="Press this button after you finish uploading.",
        disabled=False,
        button_style="info",
        tooltip="",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )



    make_recon_boxes_button.on_click(make_recon_boxes)

    method_output = Output()
    output0 = Output()

    #################### START Recon #######################
    def set_options_and_run_align(self):
        self.icon = "fas fa-cog fa-spin fa-lg"
        self.description = (
            "Setting options and loading data into memory for reconstruction."
        )

        try:
            self.description = "Reconstructing your data."
            # for i in len()
            self.button_style = "success"
            self.icon = "fa-check-square"
            self.description = "Finished reconstruction."
        except:
            self.button_style = "warning"
            self.icon = "exclamation-triangle"
            self.description = "Something went wrong."




    start_recon_button = Button(
        description="After finishing your edits above, click here to start reconstruction.",
        disabled=False,
        button_style="info",
        tooltip="",
        icon="",
        layout=Layout(width="auto", justify_content="center"),
    )
    recon_tab = Tab()
    recon_tab_vbox = VBox([make_recon_boxes_button, recon_tab])
    recon_dashboard_tabs = [recon_files, plot_vbox, recon_tab_vbox, main_handler.out]
    recon_dashboard_titles = ['Upload', 'Plot', 'Reconstruction','Log']
    recon_dashboard = Tab(titles=recon_dashboard_titles)
    recon_dashboard.children = recon_dashboard_tabs

    return recon_dashboard
