import tomopyui.widgets.meta as meta
import functools
from ipywidgets import *


def create_dashboard():

    file_import = meta.Import()
    import_widgets = [
        [file_import.filechooser],
        file_import.angles_textboxes,
        file_import.opts_checkboxes,
    ]
    import_widgets = [item for sublist in import_widgets for item in sublist]
    import_tab = HBox(import_widgets)

    center_tab_obj = meta.Center(file_import)
    prep_tab_obj = meta.Prep(file_import)
    recon_tab_obj = meta.Recon(file_import)
    align_tab_obj = meta.Align(file_import)

    dashboard_tabs = [
        import_tab,
        center_tab_obj.center_tab,
        align_tab_obj.tab,
        recon_tab_obj.tab,
        file_import.log_handler.out,
    ]
    dashboard_titles = ["Import", "Center", "Align", "Reconstruct", "Log"]
    dashboard = Tab(titles=dashboard_titles)
    dashboard.children = dashboard_tabs
    return (
        dashboard,
        file_import,
        center_tab_obj,
        prep_tab_obj,
        align_tab_obj,
        recon_tab_obj,
    )