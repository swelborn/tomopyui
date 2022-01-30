from ipywidgets import *
from tomopyui.widgets.helpers import import_module_set_env
import multiprocessing
from tomopyui.widgets.imports import Import_SSRL62
from tomopyui.widgets.center import Center
from tomopyui.widgets.analysis import Align, Recon
from tomopyui.widgets.dataexplorer import DataExplorerTab

# checks if cupy is installed. if not, disable cuda and certain gui aspects
# TODO: can put this somewhere else
cuda_import_dict = {"cupy": "cuda_enabled"}
import_module_set_env(cuda_import_dict)

# checks how many cpus available for compute on CPU
# TODO: can later add a bounded textbox for amount of CPUs user wants to use
# for reconstruction. right now defaults to all cores being used.
os.environ["num_cpu_cores"] = str(multiprocessing.cpu_count())


def create_dashboard():
    """
    This is the function to open the app in a jupyter notebook. In jupyter,
    run the following commands:

    .. code-block:: python

        %matplotlib ipympl
        import tomopyui.widgets.main as main

        dashboard, file_import, center, prep, align, recon = main.create_dashboard()
        dashboard

    """

    file_import = Import_SSRL62()
    center = Center(file_import)
    align = Align(file_import, center)
    recon = Recon(file_import, center)
    dataexplorer = DataExplorerTab(align, recon)

    for checkbox in (
        align.astra_cuda_methods_checkboxes + recon.astra_cuda_methods_checkboxes
    ):
        if os.environ["cuda_enabled"] == "True":
            checkbox.disabled = False
        else:
            checkbox.disabled = True
    for checkbox in align.tomopy_methods_checkboxes:
        if os.environ["cuda_enabled"] == "True":
            checkbox.disabled = True
        else:
            checkbox.disabled = False

    dashboard_tabs = [
        file_import.tab,
        center.tab,
        align.tab,
        recon.tab,
        dataexplorer.tab,
        file_import.log_handler.out,
    ]

    dashboard_titles = [
        "Import",
        "Center",
        "Align",
        "Reconstruct",
        "Data Explorer",
        "Log",
    ]

    dashboard = Tab(titles=dashboard_titles)
    dashboard.children = dashboard_tabs
    # workaround for nested bqplot issue
    def update_dashboard(change):
        dashboard.children = dashboard_tabs
        with dashboard_output:
            dashboard_output.clear_output(wait=True)
            display(dashboard)

    accordions = [
        file_import.raw_accordion,
        file_import.prenorm_accordion,
        center.manual_center_accordion,
        align.plotter_accordion,
        recon.plotter_accordion,
        dataexplorer.analysis_browser_accordion,
        dataexplorer.recent_alignment_accordion,
        dataexplorer.recent_recon_accordion,
    ]

    [
        accordion.observe(update_dashboard, names="selected_index")
        for accordion in accordions
    ]

    dashboard.observe(update_dashboard, names="selected_index")
    dashboard_output = Output()
    with dashboard_output:
        display(dashboard)

    return (dashboard_output, file_import, center, align, recon, dataexplorer)
