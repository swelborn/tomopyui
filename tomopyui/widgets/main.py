import tomopyui.widgets.meta as meta
from ipywidgets import *


def create_dashboard():
    '''
    This is the function to open the app in a jupyter notebook. In jupyter,
    run the following commands:

    .. code-block:: python

        %matplotlib ipympl
        import tomopyui.widgets.main as main

        dashboard, file_import, center, prep, align, recon = main.create_dashboard()
        dashboard

    '''

    file_import = meta.Import()
    center_tab_obj = meta.Center(file_import)
    prep_tab_obj = meta.Prep(file_import)
    recon_tab_obj = meta.Recon(file_import, center_tab_obj)
    align_tab_obj = meta.Align(file_import, center_tab_obj)

    dashboard_tabs = [
        file_import.tab,
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