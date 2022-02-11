# TODO: reimplement this
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui.widgets.plot import BqImPlotter_Import, BqImPlotter_DataExplorer
from tomopyui.backend.io import Projections_Prenormalized
from tomopyui.backend.io import load_metadata, metadata_to_DataFrame
from tomopyui.widgets.analysis import Align, Recon
from ipyfilechooser import FileChooser
import pathlib
from tomopyui._sharedvars import *
import numpy as np


class DataExplorerTab:
    def __init__(self, align, recon):
        self.create_tab(align, recon)

    def create_tab(self, align, recon):
        self.align = RecentAlignExplorer(align)
        self.align.create_app()
        self.recon = RecentReconExplorer(recon)
        self.recon.create_app()
        self.any = AnalysisExplorer()
        self.any.create_app()
        self.analysis_browser_accordion = Accordion(
            children=[self.any.app],
            selected_index=None,
            titles=("Plot Any Analysis",),
        )
        self.recent_alignment_accordion = Accordion(
            children=[self.align.app],
            selected_index=None,
            titles=("Plot Recent Alignments",),
        )
        self.recent_recon_accordion = Accordion(
            children=[self.recon.app],
            selected_index=None,
            titles=("Plot Recent Reconstructions",),
        )
        self.tab = VBox(
            children=[
                self.analysis_browser_accordion,
                self.recent_alignment_accordion,
                self.recent_recon_accordion,
            ]
        )


class DataExplorerBase(ABC):
    def __init__(self):
        self.metadata = None
        self.plotter_initial = BqImPlotter_DataExplorer()
        self.plotter_initial.create_app()
        self.plotter_analyzed = BqImPlotter_DataExplorer(self.plotter_initial)
        self.plotter_analyzed.create_app()
        self.projections = Projections_Prenormalized()
        self.analyzed_projections = Projections_Prenormalized()

    @abstractmethod
    def create_app(self):
        ...


class AnalysisExplorer(DataExplorerBase):
    def __init__(self):
        super().__init__()
        self.filebrowser = Filebrowser()
        self.filebrowser.create_app()
        self.filebrowser.load_data_button.on_click(self.load_data_from_filebrowser)

    def load_data_from_filebrowser(self, change):
        self.filebrowser.load_data_button.icon = "fas fa-cog fa-spin fa-lg"
        self.filebrowser.load_data_button.button_style = "info"
        metadata = {}
        self.projections.filedir = self.filebrowser.root_filedir
        self.projections.data = np.load(
            self.projections.filedir / "normalized_projections.npy"
        )
        self.projections._check_downsampled_data()
        self.plotter_initial.plot(
            self.projections.prj_imgs,
            self.projections.filedir,
            io=self.projections,
            precomputed_hists=self.projections.hists,
        )
        self.plotter_analyzed.plot(
            np.load(self.filebrowser.selected_data_fullpath),
            self.filebrowser.selected_data_fullpath.parent,
        )
        self.filebrowser.load_data_button.icon = "fa-check-square"
        self.filebrowser.load_data_button.button_style = "success"

    def create_app(self):
        plots = HBox(
            [self.plotter_initial.app, self.plotter_analyzed.app],
            layout=Layout(justify_content="center"),
        )
        self.app = VBox([self.filebrowser.app, plots])


class RecentAnalysisExplorer(DataExplorerBase):
    def __init__(self, analysis):
        super().__init__()
        self.load_run_list_button = Button(
            icon="download",
            button_style="info",
            layout=Layout(width="auto"),
        )
        self.load_run_list_button.on_click(self._load_run_list_on_click)
        self.run_list_selector = Select(
            options=[],
            rows=5,
            disabled=False,
            style=extend_description_style,
            layout=Layout(justify_content="center"),
        )
        self.run_list_selector.observe(self.choose_file_to_plot, names="value")

    def _load_run_list_on_click(self, change):
        self.load_run_list_button.button_style = "info"
        self.load_run_list_button.icon = "fas fa-cog fa-spin fa-lg"
        self.load_run_list_button.description = "Importing run list."
        # creates a list from the keys in pythonic way
        # from https://stackoverflow.com/questions/11399384/extract-all-keys-from-a-list-of-dictionaries
        # don't know how it works
        self.run_list_selector.options = list(
            set().union(*(d.keys() for d in self.analysis.run_list))
        )
        self.load_run_list_button.button_style = "success"
        self.load_run_list_button.icon = "fa-check-square"
        self.load_run_list_button.description = "Finished importing run list."

    def find_file_in_metadata(self, filedir):
        for run in range(len(self.analysis.run_list)):
            if filedir in self.analysis.run_list[run]:
                metadata = {}
                metadata["filedir"] = self.analysis.run_list[run][filedir][
                    "parent_filedir"
                ]
                metadata["filename"] = self.analysis.run_list[run][filedir][
                    "parent_filename"
                ]
                metadata["angle_start"] = self.analysis.run_list[run][filedir][
                    "angle_start"
                ]
                metadata["angle_end"] = self.analysis.run_list[run][filedir][
                    "angle_end"
                ]
                self.imagestacks[0] = TomoData(metadata=metadata).prj_imgs
                metadata["filedir"] = self.analysis.run_list[run][filedir]["savedir"]
                if self.obj.widget_type == "Align":
                    metadata["filename"] = "projections_after_alignment.tif"
                else:
                    metadata["filename"] = "recon.tif"
                self.imagestacks[1] = TomoData(metadata=metadata).prj_imgs
                self._create_image_app()

    def choose_file_to_plot(self, change):
        self.find_file_in_metadata(change.new)

    def create_app(self):
        plots = HBox(
            [self.plotter_initial.app, self.plotter_analyzed.app],
            layout=Layout(justify_content="center"),
        )
        self.app = VBox([self.load_run_list_button, self.run_list_selector, plots])


class RecentAlignExplorer(RecentAnalysisExplorer):
    def __init__(self, align: Align):
        super().__init__(align)
        self.analysis = align
        self.run_list_selector.description = "Alignments:"
        self.load_run_list_button.description = "Load alignments from this session."
        self.create_app()


class RecentReconExplorer(RecentAnalysisExplorer):
    def __init__(self, recon: Recon):
        super().__init__(recon)
        self.analysis = recon
        self.run_list_selector.description = "Reconstructions:"
        self.load_run_list_button.description = (
            "Load reconstructions from this session."
        )
        self.create_app()


class Filebrowser:
    def __init__(self):

        # parent directory filechooser
        self.orig_data_fc = FileChooser()
        self.orig_data_fc.register_callback(self.update_orig_data_folder)
        self.fc_label = Label("Original Data", layout=Layout(justify_content="Center"))

        # subdirectory selector
        self.subdir_list = []
        self.subdir_label = Label(
            "Analysis Directories", layout=Layout(justify_content="Center")
        )
        self.subdir_selector = Select(options=self.subdir_list, rows=5, disabled=False)
        self.subdir_selector.observe(self.populate_methods_list, names="value")
        self.selected_subdir = None

        # method selector
        self.methods_label = Label("Methods", layout=Layout(justify_content="Center"))
        self.methods_list = []
        self.methods_selector = Select(
            options=self.methods_list, rows=5, disabled=False
        )
        self.methods_selector.observe(self.populate_data_list, names="value")
        self.selected_method = None

        # data selector
        self.data_label = Label("Data", layout=Layout(justify_content="Center"))
        self.data_list = []
        self.data_selector = Select(options=self.data_list, rows=5, disabled=False)
        self.data_selector.observe(self.set_data_filename, names="value")
        self.allowed_extensions = (".npy", ".tif", ".tiff")
        self.selected_data_filename = None
        self.selected_data_ftype = None
        self.selected_analysis_type = None
        self.options_metadata_table_output = Output()

        # load data button
        self.load_data_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="info",
            layout=Layout(width="75px", height="86px"),
        )

    def populate_subdirs_list(self):
        self.subdir_list = [
            pathlib.Path(f) for f in os.scandir(self.root_filedir) if f.is_dir()
        ]
        self.subdir_list = [
            subdir.parts[-1]
            for subdir in self.subdir_list
            if any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.subdir_selector.options = self.subdir_list

    def update_orig_data_folder(self):
        self.root_filedir = pathlib.Path(self.orig_data_fc.selected_path)
        self.populate_subdirs_list()
        self.methods_selector.options = []

    def populate_methods_list(self, change):
        self.selected_subdir = pathlib.Path(self.root_filedir) / change.new
        self.methods_list = [
            pathlib.Path(f) for f in os.scandir(self.selected_subdir) if f.is_dir()
        ]
        self.methods_list = [
            subdir.parts[-1]
            for subdir in self.methods_list
            if not any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.methods_selector.options = self.methods_list

    def populate_data_list(self, change):
        if change.new is not None:
            self.selected_method = (
                pathlib.Path(self.root_filedir) / self.selected_subdir / change.new
            )
            self.file_list = [
                pathlib.Path(f)
                for f in os.scandir(self.selected_method)
                if not f.is_dir()
            ]
            self.data_list = [
                file.name
                for file in self.file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            self.data_selector.options = self.data_list
            self.load_metadata()
        else:
            self.data_selector.options = []

    def set_data_filename(self, change):
        self.selected_data_filename = change.new
        self.selected_data_fullpath = self.selected_method / self.selected_data_filename
        self.selected_data_ftype = pathlib.Path(self.selected_data_filename).suffix
        if "recon" in pathlib.Path(self.selected_subdir).name:
            self.selected_analysis_type = "recon"
        elif "align" in pathlib.Path(self.selected_subdir).name:
            self.selected_analysis_type = "align"

    def load_metadata(self):
        self.metadata_file = [
            self.selected_method / file.name
            for file in self.file_list
            if "metadata.json" in file.name
        ]
        if self.metadata_file != []:
            self.metadata = load_metadata(fullpath=self.metadata_file[0])
            self.options_table = metadata_to_DataFrame(self.metadata)
            with self.options_metadata_table_output:
                self.options_metadata_table_output.clear_output(wait=True)
                display(self.options_table)

    def create_app(self):
        fc = VBox([self.fc_label, self.orig_data_fc])
        subdir = VBox([self.subdir_label, self.subdir_selector])
        methods = VBox([self.methods_label, self.methods_selector])
        data = VBox([self.data_label, self.data_selector])
        button = VBox(
            [
                Label("Upload", layout=Layout(justify_content="center")),
                self.load_data_button,
            ]
        )
        top_hb = HBox(
            [fc, subdir, methods, data, button],
            layout=Layout(justify_content="center"),
            align_items="stretch",
        )
        box = VBox(
            [top_hb, self.options_metadata_table_output],
            layout=Layout(justify_content="center", align_items="center"),
        )
        self.app = box
