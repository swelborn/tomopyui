import tomopyui.widgets.meta as meta
import functools
from ipywidgets import *

def create_dashboard():

    file_import = meta.Import()
    import_widgets = [[file_import.filechooser], file_import.angles_textboxes, file_import.opts_checkboxes]
    import_widgets = [item for sublist in import_widgets for item in sublist]
    import_tab = HBox(import_widgets)

    prep_tab_obj = meta.Prep(file_import)
    recon_tab_obj = meta.Recon(file_import)
    align_tab_obj = meta.Align(file_import)

    dashboard_tabs = [import_tab, align_tab_obj.alignment_tab, recon_tab_obj.recon_tab, file_import.log_handler.out]
    dashboard_titles = ["Import", "Align", "Reconstruct", "Log"]
    dashboard = Tab(titles=dashboard_titles)
    dashboard.children = dashboard_tabs
    return dashboard, file_import, prep_tab_obj, align_tab_obj, recon_tab_obj

# def make_prep_tab(self):
#     self.icon = "fas fa-cog fa-spin fa-lg"
#     self.button_style = "info"
#     prep = meta.Prep(a)

#     return prep

# make_prep_tab_button = Button(
#     description="Click to normalize after selecting your data in the import tab.",
#     disabled=False,
#     button_style="info",
#     tooltip="",
#     icon="",
#     layout=Layout(width="auto", justify_content="center"),
# )
# make_prep_tab_button.observe(make_prep_tab)

# def make_alignment_tab():
#     self.icon = "fas fa-cog fa-spin fa-lg"
#     self.button_style = "info"
#     alignment = meta.Align(a)

#     return alignment

# make_alignment_tab_button = Button(
#     description="Click to align after selecting your data in the import tab.",
#     disabled=False,
#     button_style="info",
#     tooltip="",
#     icon="",
#     layout=Layout(width="auto", justify_content="center"),
# )
# make_alignment_tab_button.observe(make_alignment_tab)

# def make_recon_tab():
#     self.icon = "fas fa-cog fa-spin fa-lg"
#     self.button_style = "info"
#     reconstruct = meta.Recon(a)
    
#     return reconstruct

# make_recon_tab_button = Button(
#     description="Click to reconstruct your data.",
#     disabled=False,
#     button_style="info",
#     tooltip="",
#     icon="",
#     layout=Layout(width="auto", justify_content="center"),
# )
# make_recon_tab_button.observe(make_recon_tab)








# from ipywidgets import *

# import tomopy.data.tomodata as td
# from .plot_aligned_data import plot_aligned_data
# from .multiple_recon import make_recon_tab







# # Adding recon tabs
# extend_description_style = {"description_width": "auto"}
# dashboard_tabs = []
# dashboard_tabs_titles = []

# def make_dashboard_tabs(self):
#     self.icon = "fas fa-cog fa-spin fa-lg"
#     self.button_style = "info"
#     dashboard_tabs = []
#     dashboard_tabs_titles = []
#     for key in reconmetadata["tomo"]:
#         if "fpath" in reconmetadata["tomo"][key]:
#             dashboard_tabs.append(make_recon_tab(reconmetadata["tomo"][key]))
#             if "fname" in reconmetadata["tomo"][key]:
#                 dashboard_tabs_titles.append(reconmetadata["tomo"][key]["fname"])
#             else:
#                 dashboard_tabs_titles.append(reconmetadata["tomo"][key]["fpath"])
#     dashboard_tabs.children = recon_dashboard
#     dashboard_tabs.titles = recon_dashboard_titles
#     self.icon = "fa-check-square"
#     self.button_style = "success"
#     self.description = "Make your edits to the reconstruction options below. Click this again to upload more data."
#     recon_tab_vbox.children = recon_tab_vbox.children + (start_recon_button,)

# make_dashboard_tabs_button = Button(
#     description="Press this button after you finish uploading.",
#     disabled=False,
#     button_style="info",
#     tooltip="",
#     icon="",
#     layout=Layout(width="auto", justify_content="center"),
# )

# make_dashboard_tabs_button.on_click(make_dashboard_tabs)

# method_output = Output()
# output0 = Output()

# #################### START Recon #######################
# def set_options_and_run_align(self):
#     self.icon = "fas fa-cog fa-spin fa-lg"
#     self.description = (
#         "Setting options and loading data into memory for reconstruction."
#     )

#     try:
#         self.description = "Reconstructing your data."
#         # for i in len()
#         self.button_style = "success"
#         self.icon = "fa-check-square"
#         self.description = "Finished reconstruction."
#     except:
#         self.button_style = "warning"
#         self.icon = "exclamation-triangle"
#         self.description = "Something went wrong."

# start_recon_button = Button(
#     description="After finishing your edits above, click here to start reconstruction.",
#     disabled=False,
#     button_style="info",
#     tooltip="",
#     icon="",
#     layout=Layout(width="auto", justify_content="center"),
# )

# return recon_dashboard
