import pathlib
from abc import ABC, abstractmethod

import dxchange
import numpy as np
from ipyfilechooser import FileChooser
from ipywidgets import *

from tomopyui._sharedvars import extend_description_style
from tomopyui.backend.io import Projections_Prenormalized
from tomopyui.widgets.analysis import Align, Recon
from tomopyui.widgets.view import (
    BqImViewer_Projections_Child,
    BqImViewer_Projections_Parent,
)


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
            selected_index=0,
            titles=("Plot Any Analysis",),
        )
        # self.recent_alignment_accordion = Accordion(
        #     children=[self.align.app],
        #     selected_index=None,
        #     titles=("Plot Recent Alignments",),
        # )
        # self.recent_recon_accordion = Accordion(
        #     children=[self.recon.app],
        #     selected_index=None,
        #     titles=("Plot Recent Reconstructions",),
        # )
        self.tab = VBox(
            children=[
                self.analysis_browser_accordion,
                # self.recent_alignment_accordion,
                # self.recent_recon_accordion,
            ]
        )


class DataExplorerBase(ABC):
    def __init__(self):
        self.metadata = None
        self.viewer_initial = BqImViewer_Projections_Parent()
        self.viewer_initial.create_app()
        self.viewer_analyzed = BqImViewer_Projections_Child(self.viewer_initial)
        self.viewer_analyzed.create_app()
        self.projections = Projections_Prenormalized()
        self.analyzed_projections = Projections_Prenormalized()

    @abstractmethod
    def create_app(self): ...


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
        if ".npy" in self.filebrowser.selected_data_filepath.name:
            self.analyzed_projections.data = np.load(
                self.filebrowser.selected_data_filepath
            )
        elif ".tif" in self.filebrowser.selected_data_filepath.name:
            self.analyzed_projections.data = np.array(
                dxchange.reader.read_tiff(
                    self.filebrowser.selected_data_filepath
                ).astype(np.float32)
            )
        self.analyzed_projections.filedir = (
            self.filebrowser.selected_data_filepath.parent
        )
        self.projections._check_downsampled_data()
        self.viewer_initial.plot(self.projections)
        self.analyzed_projections._check_downsampled_data()
        self.viewer_analyzed.plot(self.analyzed_projections)
        self.filebrowser.load_data_button.icon = "fa-check-square"
        self.filebrowser.load_data_button.button_style = "success"

    def create_app(self):
        plots = HBox(
            [self.viewer_initial.app, self.viewer_analyzed.app],
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
                self.imagess[0] = TomoData(metadata=metadata).prj_imgs
                metadata["filedir"] = self.analysis.run_list[run][filedir]["savedir"]
                if self.obj.widget_type == "Align":
                    metadata["filename"] = "projections_after_alignment.tif"
                else:
                    metadata["filename"] = "recon.tif"
                self.imagess[1] = TomoData(metadata=metadata).prj_imgs
                self._create_image_app()

    def choose_file_to_plot(self, change):
        self.find_file_in_metadata(change.new)

    def create_app(self):
        plots = HBox(
            [self.viewer_initial.app, self.viewer_analyzed.app],
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
        self.orig_data_fc.show_only_dirs = True
        self.orig_data_fc.register_callback(self.update_orig_data_folder)
        self.fc_label = Label("Original Data", layout=Layout(justify_content="Center"))
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn\your\folder\with\normalized\projections",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )

        # subdirectory selector
        self.subdir_list = []
        self.subdir_label = Label(
            "Analysis Directories", layout=Layout(justify_content="Center")
        )
        self.subdir_selector = Select(options=self.subdir_list, rows=5, disabled=False)
        self.subdir_selector.observe(self.populate_methods_list, names="value")
        self.selected_subdir = None

        # method selector
        self.methods_list = []
        self.methods_label = Label("Methods", layout=Layout(justify_content="Center"))
        self.methods_selector = Select(
            options=self.methods_list, rows=5, disabled=False
        )
        self.methods_selector.observe(self.populate_data_list, names="value")
        self.selected_method = None

        # data selector
        self.data_list = []
        self.data_label = Label("Data", layout=Layout(justify_content="Center"))
        self.data_selector = Select(options=self.data_list, rows=5, disabled=False)
        self.data_selector.observe(self.set_data_filename, names="value")
        self.allowed_extensions = (".npy", ".tif", ".tiff")
        self.options_metadata_table_output = Output()

        # load data button
        self.load_data_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="info",
            layout=Layout(width="75px", height="86px"),
        )

    def _init_lists(self):
        self.data_list = []
        self.selected_data_filename = None
        self.selected_data_ftype = None
        self.selected_subdir = None
        self.methods_list = []
        self.selected_method = None
        self.subdir_list = []
        self.selected_analysis_type = None
        self.populate_subdirs_list()

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        try:
            self.orig_data_fc.reset(path=path)
        except Exception as e:
            with self.options_metadata_table_output:
                self.options_metadata_table_output.clear_output(wait=True)
                print(f"{e}")
            return
        else:
            self.root_filedir = path
            self._init_lists()

    def update_orig_data_folder(self):
        self.root_filedir = pathlib.Path(self.orig_data_fc.selected_path)
        self.quick_path_search.value = str(self.root_filedir)

    def populate_subdirs_list(self):
        self.subdir_list = [
            pathlib.Path(f) for f in os.scandir(self.root_filedir) if f.is_dir()
        ]
        self.subdir_list = [
            subdir.parts[-1]
            for subdir in self.subdir_list
            if any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        if self.subdir_list != []:
            self.subdir_selector.options = self.subdir_list
            self.subdir_selector.value = self.subdir_selector.options[0]
            self.populate_methods_list()
        else:
            self.data_selector.options = []
            self.methods_selector.options = []
            self.subdir_selector.options = []

    def populate_methods_list(self, *args):
        if self.subdir_selector.options != tuple():
            self.selected_subdir = (
                pathlib.Path(self.root_filedir) / self.subdir_selector.value
            )
            self.methods_list = [
                pathlib.Path(f) for f in os.scandir(self.selected_subdir) if f.is_dir()
            ]
            self.methods_list = [
                subdir.parts[-1]
                for subdir in self.methods_list
                if not any(x in subdir.parts[-1] for x in ("-align", "-recon"))
            ]
            if self.methods_list != []:
                self.methods_selector.options = self.methods_list
                self.methods_selector.value = self.methods_list[0]
                self.populate_data_list()
            else:
                self.data_selector.options = []
                self.methods_selector.options = []

    def populate_data_list(self, *args):
        if self.methods_selector.options != tuple():
            self.selected_method = (
                pathlib.Path(self.root_filedir)
                / self.selected_subdir
                / self.methods_selector.value
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
            if self.data_list != []:
                self.data_selector.options = self.data_list
                self.load_metadata()
            else:
                self.data_selector.options = []

    def set_data_filename(self, change):
        if self.data_selector.options != tuple():
            self.selected_data_filename = change.new
            self.selected_data_filepath = (
                self.selected_method / self.selected_data_filename
            )
            self.selected_data_ftype = pathlib.Path(self.selected_data_filename).suffix
            if "recon" in pathlib.Path(self.selected_subdir).name:
                self.selected_analysis_type = "recon"
            elif "align" in pathlib.Path(self.selected_subdir).name:
                self.selected_analysis_type = "align"

    def load_metadata(self):
        self.imported_metadata = False
        self.metadata_file = [
            self.selected_method / file.name
            for file in self.file_list
            if "recon_metadata.json" in file.name or "align_metadata.json" in file.name
        ]

        if self.metadata_file != []:
            self.metadata = load_metadata(filepath=self.metadata_file[0])
            self.options_table = metadata_to_DataFrame(self.metadata)
            with self.options_metadata_table_output:
                self.options_metadata_table_output.clear_output(wait=True)
                display(self.options_table)
                self.imported_metadata = True

    def create_app(self):
        quickpath = VBox(
            [
                widgets.Label("Quick path search:"),
                self.quick_path_search,
            ],
            layout=Layout(align_items="center"),
        )
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
            [quickpath, top_hb, self.options_metadata_table_output],
            layout=Layout(justify_content="center", align_items="center"),
        )
        self.app = box
