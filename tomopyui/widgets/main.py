from ipywidgets import *
from tomopyui.widgets._shared.helpers import import_module_set_env
import tomopyui.widgets.meta as meta
import multiprocessing

# checks if cupy is installed. if not, disable cuda and certain gui aspects
# TODO: can put this somewhere else (in meta?)
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

    file_import = meta.Import()
    center = meta.Center(file_import)
    align = meta.Align(file_import, center)
    recon = meta.Recon(file_import, center)
    dataexplorer = meta.DataExplorerTab(align, recon)
    dataexplorer.create_data_explorer_tab()

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
        center.center_tab,
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

    return (
        dashboard,
        file_import,
        center,
        align,
        recon,
    )
