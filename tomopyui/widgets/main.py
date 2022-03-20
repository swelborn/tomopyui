import multiprocessing
from ipywidgets import *
from tomopyui.widgets.helpers import import_module_set_env
from tomopyui.widgets.imports import Import_SSRL62C, Import_ALS832, Import_APS
from tomopyui.widgets.center import Center
from tomopyui.widgets.analysis import Align, Recon
from tomopyui.widgets.dataexplorer import DataExplorerTab
from tomopyui.widgets.prep import Prep

# checks if cupy is installed. if not, disable cuda and certain gui aspects
# TODO: can put this somewhere else
cuda_import_dict = {"cupy": "cuda_enabled"}
import_module_set_env(cuda_import_dict)

# checks how many cpus available for compute on CPU
# TODO: can later add a bounded textbox for amount of CPUs user wants to use
# for reconstruction. right now defaults to all cores being used.
os.environ["num_cpu_cores"] = str(multiprocessing.cpu_count())


def create_dashboard(institution: str):
    """
    This is the function to open the app in a jupyter notebook. In jupyter,
    run the following commands:

    .. code-block:: python

        %matplotlib ipympl
        import tomopyui.widgets.main as main

        (
            dashboard_output,
            dashboard,
            file_import,
            prep,
            center,
            align,
            recon,
            dataexplorer,
        ) = main.create_dashboard(
            "ALS_832"
        )  # can be "SSRL_62C", "ALS_832", "APS"
        dashboard

    """
    if institution == "ALS_832":
        file_import = Import_ALS832()
    if institution == "SSRL_62C":
        file_import = Import_SSRL62C()
    if institution == "APS":
        file_import = Import_APS()
    prep = Prep(file_import)
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
        prep.tab,
        center.tab,
        align.tab,
        recon.tab,
        dataexplorer.tab,
        file_import.log_handler.out,
    ]

    dashboard_titles = [
        "Import",
        "Prep",
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
        prep.viewer_accordion,
        center.manual_center_accordion,
        align.viewer_accordion,
        recon.viewer_accordion,
        dataexplorer.analysis_browser_accordion,
        # dataexplorer.recent_alignment_accordion,
        # dataexplorer.recent_recon_accordion,
    ]

    [
        accordion.observe(update_dashboard, names="selected_index")
        for accordion in accordions
    ]

    dashboard.observe(update_dashboard, names="selected_index")
    dashboard_output = Output()
    with dashboard_output:
        display(dashboard)

    return (
        dashboard_output,
        dashboard,
        file_import,
        prep,
        center,
        align,
        recon,
        dataexplorer,
    )
