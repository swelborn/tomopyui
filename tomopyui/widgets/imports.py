from tomopyui.widgets._import import import_helpers
from tomopyui.widgets._shared import helpers
from ipyfilechooser import FileChooser
import logging
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui._sharedvars import *
import time
from tomopyui.widgets.plot import BqImPlotter_ImportRaw
from tomopyui.backend.io import RawProjectionsXRM_SSRL62
import pathlib


class ImportBase(ABC):
    def __init__(self):

        # Init textboxes
        self.angle_start = -90.0
        self.angle_end = 90.0
        self.prj_shape = None
        self.prj_range_x = (0, 100)
        self.prj_range_y = (0, 100)
        self.prj_range_z = (0, 100)
        self.tomo = None
        self.angles_textboxes = import_helpers.create_angles_textboxes(self)
        self.prenorm_projections = Projections_Prenormalized()
        self.projections = self.prenorm_projections

        # Init filechooser
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_file_information)
        self.filechooser.title = "Import Normalized Tomogram:"
        self.wd = None

        # Init filechooser for align metadata
        self.fpath_align = None
        self.fname_align = None
        self.filechooser_align = FileChooser()
        self.filechooser_align.register_callback(self.update_file_information_align)
        self.filechooser_align.title = "Import Alignment Metadata:"

        # Init filechooser for recon metadata
        self.fpath_recon = None
        self.fname_recon = None
        self.filechooser_recon = FileChooser()
        self.filechooser_recon.register_callback(self.update_file_information_recon)
        self.filechooser_recon.title = "Import Reconstruction Metadata:"

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

        # Init metadata
        self.metadata = {}
        self.set_metadata()

    def set_wd(self, wd):
        """
        Sets the current working directory of `Import` class and changes the
        current directory to it.
        """
        self.wd = wd
        os.chdir(wd)

    def set_metadata(self):
        """
        Sets relevant metadata for `Import`
        """
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta,
            "prj_range_x": self.prj_range_x,
            "prj_range_y": self.prj_range_y,
        }

    def update_file_information(self):
        """
        Callback for `Import`.filechooser.
        """
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_wd(self.fpath)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.set_metadata()
        self.get_prj_shape()
        self.set_prj_ranges()
        self.set_metadata()

    def update_file_information_align(self):
        """
        Callback for filechooser_align.
        """
        self.fpath_align = self.filechooser_align.selected_path
        self.fname_align = self.filechooser_align.selected_filename
        self.set_metadata()

    def update_file_information_recon(self):
        """
        Callback for filechooser_recon.
        """
        self.fpath_recon = self.filechooser_recon.selected_path
        self.fname_recon = self.filechooser_recon.selected_filename
        self.set_metadata()

    @abstractmethod
    def make_tab(self):
        ...


class Import_SSRL62(ImportBase):
    def __init__(self):
        super().__init__()
        # Init normalizing folder chooser
        self.fpath_raw = None
        self.fname_raw = None
        self.filechooser_raw = FileChooser()
        self.filechooser_raw.register_callback(self.update_raw_file_information)
        self.filechooser_raw.title = "Import Raw XRM Folder"
        self.angles_from_filenames = True
        self._init_widgets()
        self.raw_projections = RawProjectionsXRM_SSRL62()
        self.raw_projections.set_options_from_frontend(self)
        self.make_tab()

    def _init_widgets(self):
        self.metadata_table_output = Output()
        self.import_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="",
            layout=Layout(width="75px", height="86px"),
            disabled=True,
        )
        self.import_button.on_click(self.import_and_normalize_onclick)
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )
        self.quick_path_label = Label("Quick path search")
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.bq_plotter_raw = BqImPlotter_ImportRaw()
        self.bq_plotter_raw.create_app()

    def import_and_normalize_onclick(self, change):
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        self.upload_progress.value = 0
        self.upload_progress.max = self.raw_projections.pxZ + len(
            self.raw_projections.flats_ind
        )
        with self.progress_output:
            display(self.upload_progress)
        self.raw_projections.import_folder_all(self.fpath_raw)
        with self.progress_output:
            display(Label("Normalizing", layout=Layout(justify_content="center")))

        self.raw_projections.normalize_nf()
        toc = time.perf_counter()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"

        with self.progress_output:
            display(
                Label(
                    f"Import and normalization took {toc-tic:.0f}s",
                    layout=Layout(justify_content="center"),
                )
            )
        self.bq_plotter_raw.plot(self.raw_projections.prj_imgs)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.fpath_raw = path
        self.filechooser_raw.reset(path=path)
        textfiles = self.raw_projections._file_finder(path, [".txt"])
        if textfiles == []:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have any .txt files, please try another one."
                )
            return
        scan_info_filepath = (
            path / [file for file in textfiles if "ScanInfo" in file][0]
        )
        if scan_info_filepath != []:
            self.raw_projections.import_metadata(path)
            self.metadata_table = self.raw_projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
        else:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have a ScanInfo file, please try another one."
                )

    def update_raw_file_information(self):

        self.fpath_raw = pathlib.Path(self.filechooser_raw.selected_path)
        self.fname_raw = self.filechooser_raw.selected_filename
        self.set_wd(self.fpath_raw)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.raw_projections.import_metadata(self.fpath_raw)
        self.metadata_table = self.raw_projections.metadata_to_DataFrame()
        self.quick_path_search.value = str(self.fpath_raw)
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False

    def disable_raw(self):

        self.use_raw = False
        self.use_prenorm = True
        self.raw_accordion.selected_index = None
        self.prenormalized_accordion.selected_index = 0

    def enable_raw(self):

        self.use_raw = True
        self.use_prenorm = False
        self.raw_accordion.selected_index = 0
        self.prenorm_accordion.selected_index = None

    def make_tab(self):

        raw_or_norm_button_box = HBox([])

        raw_import = HBox(
            [
                VBox(
                    [
                        self.quick_path_label,
                        HBox([self.quick_path_search, self.import_button]),
                        self.filechooser_raw,
                    ],
                ),
                self.bq_plotter_raw.app,
            ]
        )

        # raw_import = HBox([item for sublist in raw_import for item in sublist])
        self.raw_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        HBox(
                            [self.progress_output],
                            layout=Layout(justify_content="center"),
                        ),
                        raw_import,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )
        norm_import = [[self.filechooser], self.angles_textboxes]
        norm_import = HBox(
            [item for sublist in norm_import for item in sublist],
            layout=Layout(justify_content="center"),
        )
        self.prenorm_accordion = Accordion(
            children=[norm_import],
            selected_index=None,
            titles=("Import Pre-normalized Data",),
        )
        self.meta_accordion = Accordion(
            children=[
                HBox(
                    [self.filechooser_align, self.filechooser_recon],
                    layout=Layout(justify_content="center"),
                )
            ],
            selected_index=None,
            titles=("Import Alignment/Reconstruction Settings",),
        )
        self.tab = VBox(
            [
                raw_import,
                norm_import,
                meta_import,
            ]
        )


class UploaderBase(ABC):
    def __init__(self):

        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )

        self.quick_path_label = Label("Quick path search")
        self.filepath = None
        self.filename = None
        self.filechooser = FileChooser()
        self.import_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="",
            layout=Layout(width="75px", height="86px"),
            disabled=True,
        )

    @abstractmethod
    def update_filechooser_from_quicksearch(self, change):
        ...

    @abstractmethod
    def update_quicksearch_from_filechooser(self):
        ...

    @abstractmethod
    def import_data(self):
        ...


class PrenormUploader(UploaderBase):
    def __init__(self, PrenormProjections, Import):
        super().__init__()
        self.projections = PrenormProjections
        self.Import = Import
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Prenormalized Data"
        self.import_button.on_click(self.import_data)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filepath = path
        self.filechooser.reset(path=path)
        if self.check_for_data():
            self.import_button.button_style = "info"
            self.import_button.disabled = False

    def update_quicksearch_from_filechooser(self):

        self.filepath = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filepath)
        if self.check_for_data():
            self.import_button.button_style = "info"
            self.import_button.disabled = False

    def import_data(self, change):
        if self.filechooser.selected_filename == "":
            self.projections.import_folder_projections(self.filepath)
        else:
            self.projections.import_file_projections(self.filepath / self.filename)

    def check_for_data(self):
        file_list = self.projections._file_finder(
            self.filepath, self.projections.allowed_extensions
        )

        if len(file_list) > 0:
            return True
        else:
            return False


class RawUploader_SSRL62(UploaderBase):
    def __init__(self, RawProjections, Import):
        super().__init__()
        self.metadata_table_output = Output()
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )
        self.plotter = BqImPlotter_ImportRaw()
        self.plotter.create_app()
        self.projections = RawProjections
        self.Import = Import
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Raw XRM Folder"
        self.import_button.on_click(self.import_data)

    def import_data(self, change):
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        self.upload_progress.value = 0
        self.upload_progress.max = self.raw_projections.pxZ + len(
            self.raw_projections.flats_ind
        )
        with self.progress_output:
            display(self.upload_progress)
        self.raw_projections.import_folder_all(self.fpath_raw)
        with self.progress_output:
            display(Label("Normalizing", layout=Layout(justify_content="center")))

        self.raw_projections.normalize_nf()
        toc = time.perf_counter()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"

        with self.progress_output:
            display(
                Label(
                    f"Import and normalization took {toc-tic:.0f}s",
                    layout=Layout(justify_content="center"),
                )
            )
        self.bq_plotter_raw.plot(self.raw_projections.prj_imgs)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filepath = path
        self.filechooser.reset(path=path)
        textfiles = self.projections._file_finder(path, [".txt"])
        if textfiles == []:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have any .txt files, please try another one."
                )
            return

        scan_info_filepath = (
            path / [file for file in textfiles if "ScanInfo" in file][0]
        )
        if scan_info_filepath != []:
            self.projections.import_metadata(path)
            self.metadata_table = self.projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
        else:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have a ScanInfo file, please try another one."
                )

    def update_quicksearch_from_filechooser(self):

        self.filepath = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filepath)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.projections.import_metadata(self.filepath)
        self.metadata_table = self.projections.metadata_to_DataFrame()

        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
