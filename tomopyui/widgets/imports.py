from tomopyui.widgets import helpers
from ipyfilechooser import FileChooser
import logging
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui._sharedvars import *
import time
from tomopyui.widgets.plot import BqImPlotter_Import
from tomopyui.backend.io import RawProjectionsXRM_SSRL62, Projections_Prenormalized
import pathlib
import functools


class ImportBase(ABC):
    """"""

    def __init__(self):

        # Init raw/prenorm button swtiches
        self.use_raw_button = Button(
            description="Click to use raw/normalized data from the Import tab.",
            button_style="info",
            layout=Layout(width="auto", height="auto", align_items="stretch"),
            # style={"font_size": "18px"},
        )
        self.use_raw_button.on_click(self.enable_raw)
        self.use_prenorm_button = Button(
            description="Click to use prenormalized data from the Import tab.",
            button_style="info",
            layout=Layout(width="auto", height="auto", align_items="stretch"),
            # style={"font_size": "18px"},
        )
        self.use_prenorm_button.on_click(self.disable_raw)

        # Init textboxes
        self.angle_start = -90.0
        self.angle_end = 90.0
        self.angles_textboxes = self.create_angles_textboxes()
        self.prenorm_projections = Projections_Prenormalized()
        self.prenorm_uploader = PrenormUploader(self)
        self.wd = None
        self.alignmeta_uploader = MetadataUploader("Import Alignment Metadata:")
        self.reconmeta_uploader = MetadataUploader("Import Reconstruction Metadata:")
        self.projections = self.prenorm_projections
        self.uploader = self.prenorm_uploader

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

        # Init metadata
        self.metadata = {}
        # self.set_metadata()

    def disable_raw(self, *args):

        self.use_raw_button.description = (
            "Click to use raw/normalized data from Import tab."
        )
        self.use_raw_button.icon = ""
        self.use_raw_button.button_style = "info"
        self.use_raw = False
        self.use_prenorm = True
        # self.raw_accordion.selected_index = None
        # self.prenorm_accordion.selected_index = 0
        self.projections = self.prenorm_projections
        self.uploader = self.prenorm_uploader
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.use_prenorm_button.icon = "fas fa-cog fa-spin fa-lg"
        self.use_prenorm_button.button_style = "info"
        self.use_prenorm_button.description = "Updating plots."
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.use_prenorm_button.icon = "fa-check-square"
        self.use_prenorm_button.button_style = "success"
        self.use_prenorm_button.description = (
            "Prenormalized data from Import tab in use for alignment/reconstruction."
        )

    def enable_raw(self, *args):
        self.use_prenorm_button.description = (
            "Click to use prenormalized data from Import tab."
        )
        self.use_prenorm_button.icon = ""
        self.use_prenorm_button.button_style = "info"
        self.use_raw = True
        self.use_prenorm = False
        # self.raw_accordion.selected_index = 0
        # self.prenorm_accordion.selected_index = None
        self.projections = self.raw_projections
        self.uploader = self.raw_uploader
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.use_raw_button.icon = "fas fa-cog fa-spin fa-lg"
        self.use_raw_button.button_style = "info"
        self.use_raw_button.description = "Updating plots."
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.use_raw_button.icon = "fa-check-square"
        self.use_raw_button.button_style = "success"
        self.use_raw_button.description = (
            "Raw/normalized data from Import tab in use for alignment/reconstruction."
        )

    def set_wd(self, wd):
        """
        Sets the current working directory of `Import` class and changes the
        current directory to it.
        """
        self.wd = wd
        os.chdir(wd)

    def create_angles_textboxes(self):
        """
        Creates textboxes for angle start/angle end.
        """

        def create_textbox(description, value, metadatakey, int=False):
            def angle_callbacks(change, key):
                self.metadata[key] = change.new
                if key == "angle_start":
                    self.angle_start = self.metadata[key]
                if key == "angle_end":
                    self.angle_end = self.metadata[key]

            textbox = FloatText(
                value=value,
                description=description,
                disabled=False,
                style=extend_description_style,
            )

            textbox.observe(
                functools.partial(angle_callbacks, key=metadatakey),
                names="value",
            )
            return textbox

        angle_start = create_textbox("Starting angle (\u00b0): ", -90, "angle_start")
        angle_end = create_textbox("Ending angle (\u00b0): ", 90, "angle_end")

        angles_textboxes = [angle_start, angle_end]
        return angles_textboxes

    @abstractmethod
    def make_tab(self):
        ...


class Import_SSRL62(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.angles_from_filenames = True
        self.raw_projections = RawProjectionsXRM_SSRL62()
        self.raw_uploader = RawUploader_SSRL62(self)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = HBox(
            [self.use_raw_button, self.use_prenorm_button],
            layout=Layout(justify_content="center"),
        )

        self.raw_import = HBox(
            [
                VBox(
                    [
                        self.raw_uploader.quick_path_label,
                        HBox(
                            [
                                self.raw_uploader.quick_path_search,
                                self.raw_uploader.import_button,
                            ]
                        ),
                        self.raw_uploader.filechooser,
                        self.raw_uploader.nm_per_px_textbox,
                    ],
                ),
                self.raw_uploader.plotter.app,
            ],
            layout=Layout(justify_content="center"),
        )

        # raw_import = HBox([item for sublist in raw_import for item in sublist])
        self.raw_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.raw_uploader.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        HBox(
                            [self.raw_uploader.progress_output],
                            layout=Layout(justify_content="center"),
                        ),
                        self.raw_import,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )
        self.norm_import = HBox(
            [
                VBox(
                    [
                        self.prenorm_uploader.quick_path_label,
                        HBox(
                            [
                                self.prenorm_uploader.quick_path_search,
                                self.prenorm_uploader.import_button,
                            ]
                        ),
                        self.prenorm_uploader.filechooser,
                        HBox(self.angles_textboxes),
                        self.prenorm_uploader.nm_per_px_textbox,
                    ],
                ),
                self.prenorm_uploader.plotter.app,
            ],
            layout=Layout(justify_content="center"),
        )

        self.prenorm_accordion = Accordion(
            children=[self.norm_import],
            selected_index=None,
            titles=("Import Prenormalized Data",),
        )

        self.meta_accordion = Accordion(
            children=[
                HBox(
                    [
                        self.alignmeta_uploader.filechooser,
                        self.reconmeta_uploader.filechooser,
                    ],
                    layout=Layout(justify_content="center"),
                )
            ],
            selected_index=None,
            titles=("Import Alignment/Reconstruction Settings",),
        )
        self.tab = VBox(
            [
                # self.switch_data_buttons,
                self.raw_accordion,
                self.prenorm_accordion,
                self.meta_accordion,
            ]
        )


class UploaderBase(ABC):
    """"""

    def __init__(self):
        self.filedir = None
        self.filename = None
        self.nm_per_px = None
        self.filechooser = FileChooser()
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_label = Label("Quick path search")
        self.import_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="",
            layout=Layout(width="75px", height="86px"),
            disabled=True,
            tooltip="Load your data into memory",
        )
        self.nm_per_px_textbox = FloatText(
            description="nm/px (for binning 1):",
            style=extend_description_style,
        )
        self.nm_per_px_textbox.observe(self.update_nm_per_px, "value")

    def update_nm_per_px(self, *args):
        self.nm_per_px = self.nm_per_px_textbox.value

    @abstractmethod
    def update_filechooser_from_quicksearch(self, change):
        ...

    @abstractmethod
    def update_quicksearch_from_filechooser(self):
        ...

    @abstractmethod
    def import_data(UploaderBase):
        ...


class MetadataUploader(UploaderBase):
    """"""

    def __init__(self, title):
        super().__init__()

        self.filedir = None
        self.filename = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = title
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn\data\metadata.json",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )

        self.quick_path_label = Label("Quick path search")
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filedir = path
        self.filechooser.reset(path=path)

    def update_quicksearch_from_filechooser(self):
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filedir / self.filename)

    def import_data(self):
        pass


class PrenormUploader(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self.projections = Import.prenorm_projections
        self.Import = Import
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import prenormalized data:"
        self.import_button.on_click(self.import_data)
        self._tmp_disable_reset = False
        self.plotter = BqImPlotter_Import()
        self.plotter.create_app()

    def update_filechooser_from_quicksearch(self, change):
        if not self._tmp_disable_reset:
            path = pathlib.Path(change.new)
            if path.is_dir():
                self.filedir = path
                self.filechooser.reset(path=self.filedir)
            elif any(x in file.name for x in self.projections.allowed_extensions):
                self.filedir = path.parent
                self.filename = path.name
            self.filechooser.reset(path=self.filedir)
            if self.check_for_data():
                self.import_button.button_style = "info"
                self.import_button.disabled = False

    def update_quicksearch_from_filechooser(self):
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self._tmp_disable_reset = True
        self.quick_path_search.value = str(self.filedir / self.filename)
        self._tmp_disable_reset = False
        if self.check_for_data():
            self.import_button.button_style = "info"
            self.import_button.disabled = False

    def import_data(self, change):
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.projections.set_options_from_frontend(self.Import, self)
        if self.filechooser.selected_filename == "":
            self.projections.import_filedir_projections(self.filedir)
        else:
            self.projections.import_file_projections(self.filedir / self.filename)
        self.projections.save_normalized_as_npy()
        self.projections.nm_per_px = self.nm_per_px_textbox.value
        self.plotter.plot(self.projections.prj_imgs, self.filedir)
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"
        self.Import.use_raw_button.icon = ""
        self.Import.use_raw_button.button_style = "info"
        self.Import.use_raw_button.description = (
            "Click to use raw/normalized data from the Import tab."
        )
        self.Import.use_prenorm_button.icon = ""
        self.Import.use_prenorm_button.button_style = "info"
        self.Import.use_prenorm_button.description = (
            "Click to use prenormalized data from the Import tab."
        )

    def check_for_data(self):
        file_list = self.projections._file_finder(
            self.filedir, self.projections.allowed_extensions
        )

        if len(file_list) > 0:
            return True
        else:
            return False


class RawUploader_SSRL62(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()
        self.projections = Import.raw_projections
        self.Import = Import
        self.import_button.on_click(self.import_data)
        self.projections.set_options_from_frontend(self.Import, self)
        self.plotter = BqImPlotter_Import()
        self.plotter.create_app()
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Choose a Raw XRM File Directory"

    def _init_widgets(self):
        self.metadata_table_output = Output()
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )

    def import_data(self, change):
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        self.upload_progress.value = 0
        self.upload_progress.max = self.projections.pxZ + len(
            self.projections.flats_ind
        )
        with self.progress_output:
            display(self.upload_progress)

        self.projections.import_filedir_all(self.filedir)
        with self.progress_output:
            display(Label("Normalizing", layout=Layout(justify_content="center")))
        self.projections.normalize_nf()
        with self.progress_output:
            display(
                Label(
                    "Saving projections as npy for faster IO",
                    layout=Layout(justify_content="center"),
                )
            )
        self.projections.save_normalized_as_npy()
        self.projections.nm_per_px = self.nm_per_px_textbox.value
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
        self.plotter.plot(self.projections.prj_imgs)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        try:
            self.filechooser.reset(path=path)
        except Exception as e:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(f"{e}")
            return
        else:
            self.filedir = path

        try:
            textfiles = self.projections._file_finder(path, [".txt"])
            assert textfiles != []
        except AssertionError:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print("No .txt files found in this directory.")
            return
        else:
            try:
                scan_info_filepath = (
                    path / [file for file in textfiles if "ScanInfo" in file][0]
                )
            except Exception:
                with self.metadata_table_output:
                    self.metadata_table_output.clear_output(wait=True)
                    print(
                        "This directory doesn't have a ScanInfo file, please try another one."
                    )
                return
        try:
            assert scan_info_filepath != []

        except Exception:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This directory doesn't have a ScanInfo file, please try another one."
                )
            return

        else:
            self.projections.import_metadata(path)
            self.metadata_table = self.projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False

    def update_quicksearch_from_filechooser(self):

        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filedir)
        # metadata must be set here in case tomodata is created (for filedir
        # import). this can be changed later.
        self.projections.import_metadata(self.filedir)
        self.metadata_table = self.projections.metadata_to_DataFrame()

        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
