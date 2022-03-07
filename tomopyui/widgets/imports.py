from tomopyui.widgets import helpers
from ipyfilechooser import FileChooser
import logging
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui._sharedvars import *
import time
from tomopyui.widgets.view import BqImViewer_Import
from tomopyui.backend.io import (
    RawProjectionsXRM_SSRL62,
    Projections_Prenormalized_SSRL62,
)
import numpy as np
import pathlib
import functools
from ipyfilechooser.errors import InvalidPathError, InvalidFileNameError
import re


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
        self.prenorm_projections = Projections_Prenormalized_SSRL62()
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
        self.Center.refresh_plots()
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
        self.Center.refresh_plots()
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
                        self.raw_uploader.file_chooser_label,
                        self.raw_uploader.quick_path_label,
                        HBox(
                            [
                                self.raw_uploader.quick_path_search,
                                self.raw_uploader.import_button,
                            ]
                        ),
                        self.raw_uploader.filechooser,
                        self.raw_uploader.energy_select_label,
                        self.raw_uploader.energy_select_multiple,
                        self.raw_uploader.energy_overwrite_textbox,
                        self.raw_uploader.save_tiff_on_import_checkbox,
                        VBox(
                            [
                                self.raw_uploader.already_uploaded_energies_label,
                                self.raw_uploader.already_uploaded_energies_select,
                            ],
                            layout=Layout(align_content="center"),
                        ),
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
                        HBox(
                            [
                                self.prenorm_uploader.renormalize_by_roi_button,
                                self.prenorm_uploader.overwrite_normalized_button,
                            ]
                        ),
                    ],
                ),
                self.prenorm_uploader.plotter.app,
            ],
            layout=Layout(justify_content="center"),
        )

        self.prenorm_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.prenorm_uploader.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        self.norm_import,
                    ]
                ),
            ],
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
        self.filedir = pathlib.Path()
        self.filename = pathlib.Path()
        self.current_pixel_size = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
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
        self.import_button.on_click(self.import_data)

        self.metadata_table_output = Output()

    def check_filepath_exists(self, path):
        self.filename = None
        self.filedir = None
        try:
            self.filechooser.reset(path=path)
        except InvalidPathError:
            try:
                self.filechooser.reset(path=path.parent)
            except InvalidPathError as e:
                raise InvalidPathError
            except InvalidFileNameError as e:
                raise InvalidFileNameError
            else:
                self.filedir = path.parent
                if (path.parent / path.name).exists():
                    self.filename = str(path.name)
        else:
            self.filedir = path

    @abstractmethod
    def update_filechooser_from_quicksearch(self, change):
        ...

    @abstractmethod
    def update_quicksearch_from_filechooser(self):
        ...

    @abstractmethod
    def import_data(self):
        ...


class MetadataUploader(UploaderBase):
    """"""

    def __init__(self, title):
        super().__init__()

        self.filedir = None
        self.filename = None
        self.filechooser = FileChooser()
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
        self.filechooser.title = "Import prenormalized data:"
        self._tmp_disable_reset = False
        self.plotter = BqImViewer_Import()
        self.plotter.create_app()
        self.renormalize_by_roi_button = Button(
            description="Click to normalize by ROI.",
            button_style="info",
            layout=Layout(width="auto", height="auto", align_items="stretch"),
            disabled=True,
        )
        self.renormalize_by_roi_button.on_click(self.renormalize_by_roi)
        self.overwrite_normalized_button = Button(
            description="Overwrite normalized_projections.npy?",
            button_style="info",
            layout=Layout(width="auto", height="auto", align_items="stretch"),
            disabled=True,
        )
        self.overwrite_normalized_button.on_click(self.overwrite_normalized)
        self.imported_metadata = False
        self.import_status_label = Label(layout=Layout(justify_content="center"))
        self.find_metadata_status_label = Label(layout=Layout(justify_content="center"))
        self.plotter.rectangle_selector_button.observe(self.rectangle_selector_on)
        self.plotter.rectangle_selector_on = False

    def renormalize_by_roi(self, change):
        self.renormalize_by_roi_button.description = "Renormalizing."
        self.renormalize_by_roi_button.icon = "fas fa-cog fa-spin fa-lg"
        self.projections.renormalize_by_roi(self)
        self.plotter.downsample_factor = 1
        self.plotter.plot(self.projections)
        self.renormalize_by_roi_button.icon = "fa-check-square"
        self.renormalize_by_roi_button.description = "Finished renormalizing."
        self.overwrite_normalized_button.disabled = False

    def overwrite_normalized(self, change):
        np.save(
            self.projections.filedir / "normalized_projections.npy",
            self.projections.data,
        )

    def rectangle_selector_on(self, change):
        time.sleep(0.1)
        if self.plotter.rectangle_selector_on:
            self.renormalize_by_roi_button.disabled = False
        else:
            self.renormalize_by_roi_button.disabled = True

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        try:
            self.check_filepath_exists(path)
        except InvalidFileNameError:
            self.find_metadata_status_label.value = (
                "Invalid file name for that directory."
            )
            return
        except InvalidPathError:
            self.find_metadata_status_label.value = "Invalid path."
            return
        else:
            self.import_button.button_style = "info"
            self.import_button.disabled = False
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.find_metadata_status_label)
            try:
                json_files = self.projections._file_finder(self.filedir, [".json"])
                assert json_files != []
            except AssertionError:
                self.find_metadata_status_label.value = (
                    "No .json files found in this directory."
                    + " Make sure you input start/end angles correctly."
                )
                for tb in self.Import.angles_textboxes:
                    tb.disabled = False
                self.imported_metadata = False
                return
            else:
                try:
                    self.import_metadata_filepath = (
                        self.filedir
                        / [file for file in json_files if "import_metadata" in file][0]
                    )
                    assert self.import_metadata_filepath != []
                except AssertionError:
                    self.find_metadata_status_label.value = (
                        "This directory has .json files but not an import_metadata.json"
                        + " file. Make sure you input start/end angles correctly."
                    )
                    for tb in self.Import.angles_textboxes:
                        tb.disabled = False
                    self.imported_metadata = False
                    return

                else:
                    self.projections.import_metadata(self)
                    self.metadata_table = self.projections.metadata_to_DataFrame()
                    with self.metadata_table_output:
                        self.metadata_table_output.clear_output(wait=True)
                        display(self.metadata_table)
                    for tb in self.Import.angles_textboxes:
                        tb.disabled = True
                    self.imported_metadata = True

    def update_quicksearch_from_filechooser(self):
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self._tmp_disable_reset = True
        self.quick_path_search.value = str(self.filedir / self.filename)
        self._tmp_disable_reset = False

    def import_data(self, change):
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        if not self.imported_metadata:
            self.angle_start = self.Import.angle_start
            self.angle_end = self.Import.angle_end
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            if self.imported_metadata:
                display(self.metadata_table)
            display(self.import_status_label)
        if self.filename == "" or self.filename is None:
            self.import_status_label.value = "Importing file directory."
            self.projections.import_filedir_projections(self)
        else:
            self.import_status_label.value = "Importing single file."
            self.projections.import_file_projections(self)
        self.import_status_label.value = "Checking for downsampled data."
        self.projections._check_downsampled_data(label=self.import_status_label)
        self.import_status_label.value = (
            "Plotting data (downsampled for viewer to 0.25x)."
        )
        self.plotter.plot(self.projections)
        toc = time.perf_counter()
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
        self.import_status_label.value = (
            f"Import, downsampling (if any), and plotting complete in ~{toc-tic:.0f}s."
        )


class RawUploader_SSRL62(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()
        self.user_overwrite_energy = False
        self.projections = Import.raw_projections
        self.Import = Import
        # self.projections.set_options_from_frontend(self.Import, self)
        self.plotter = BqImViewer_Import()
        self.plotter.create_app()
        self.filechooser.title = "Choose a Raw XRM File Directory"

    def _init_widgets(self):
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )
        self.file_chooser_label = "Find data folder"
        self.file_chooser_label = Label(
            self.file_chooser_label, style=self.header_font_style
        )
        self.energy_select_multiple = SelectMultiple(
            options=["7700.00", "7800.00", "7900.00"],
            rows=3,
            description="Energies (eV): ",
            disabled=True,
            style=extend_description_style,
        )
        self.energy_select_label = "Select energies"
        self.energy_select_label = Label(
            self.energy_select_label, style=self.header_font_style
        )
        self.energy_overwrite_textbox = FloatText(
            description="Overwrite Energy (eV): ",
            style=extend_description_style,
            disabled=True,
        )
        self.energy_overwrite_textbox.observe(self.energy_overwrite, names="value")
        self.save_tiff_on_import_checkbox = Checkbox(
            description="Save .tif on import.",
            value=False,
            style=extend_description_style,
            disabled=False,
        )
        self.already_uploaded_energies_select = Select(
            options=["7700.00", "7800.00", "7900.00"],
            rows=3,
            description="Uploaded Energies (eV): ",
            disabled=True,
            style=extend_description_style,
        )
        self.already_uploaded_energies_label = "Previously uploaded energies"
        self.already_uploaded_energies_label = Label(
            self.already_uploaded_energies_label, style=self.header_font_style
        )

    def energy_overwrite(self, *args):
        if (
            self.energy_overwrite_textbox.value
            != self.projections.energies_list_float[0]
            and self.energy_overwrite_textbox.value is not None
        ):
            self.user_input_energy_float = self.energy_overwrite_textbox.value
            self.user_input_energy_str = str(f"{self.user_input_energy_float:08.2f}")
            self.energy_select_multiple.options = [
                self.user_input_energy_str,
            ]
            self.projections.pixel_sizes = [
                self.projections.calculate_px_size(
                    self.user_input_energy_float, self.projections.binning
                )
            ]
            self.user_overwrite_energy = True

    def import_data(self, change):

        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.projections.import_filedir_all(self)
        toc = time.perf_counter()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"
        self.projections.status_label.value = (
            f"Import and normalization took {toc-tic:.0f}s"
        )
        self.projections.filedir = self.projections.energy_filedir
        self.plotter.plot(self.projections)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.check_filepath_exists(path)
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
            self.user_overwrite_energy = False
            self.projections.import_metadata(self)
            self.metadata_table = self.projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
            self.already_uploaded_energies_select.disabled = True
            self.check_energy_folders()

    def check_energy_folders(self):
        folders = [pathlib.Path(f) for f in os.scandir(self.filedir) if f.is_dir()]
        reg_exp = re.compile("\d\d\d\d\d.\d\deV")
        ener_folders = map(reg_exp.findall, [str(folder) for folder in folders])
        self.already_uploaded_energies = [
            str(folder[0][:-2]) for folder in ener_folders
        ]
        self.already_uploaded_energies_select.options = self.already_uploaded_energies
        self.already_uploaded_energies_select.disabled = False

    def update_quicksearch_from_filechooser(self):
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filedir)
        self.user_overwrite_energy = False
        self.projections.import_metadata(self)
        self.metadata_table = self.projections.metadata_to_DataFrame()
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
        self.import_button.button_style = "info"
        self.import_button.disabled = False
        if self.projections.energy_guessed:
            self.energy_overwrite_textbox.disabled = False
            self.energy_overwrite_textbox.value = self.projections.energies_list_float[
                0
            ]
        else:
            self.energy_overwrite_textbox.disabled = True
            self.energy_overwrite_textbox.value = None
