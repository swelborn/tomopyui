import copy
import os
import pathlib
import re
import time

from ipyfilechooser import FileChooser
import ipywidgets as widgets

from tomopyui.widgets.styles import (
    extend_description_style,
)
from tomopyui.backend.io import (
    Metadata,
)
from tomopyui.backend.readers.SSRL import (
    RawProjectionsTiff_SSRL62B,
    RawProjectionsXRM_SSRL62C,
)
from IPython.display import display
from .importer_base import ImportBase
from .uploader_base import UploaderBase
from tomopyui.widgets.styles import header_font_style


class Import_SSRL62B(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_SSRL62B(self)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = widgets.HBox(
            [self.use_raw_button.button, self.use_prenorm_button.button],
            layout=widgets.Layout(justify_content="center"),
        )

        # raw_import = widgets.HBox([item for sublist in raw_import for item in sublist])
        self.raw_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self.raw_uploader.metadata_table_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        widgets.HBox(
                            [self.raw_uploader.progress_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        self.raw_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )

        self.prenorm_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self.prenorm_uploader.metadata_table_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        self.prenorm_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import Prenormalized Data",),
        )

        self.tab = widgets.VBox(
            [
                self.raw_accordion,
                self.prenorm_accordion,
            ]
        )


class Import_SSRL62C(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_SSRL62C(self)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = widgets.HBox(
            [self.use_raw_button.button, self.use_prenorm_button.button],
            layout=widgets.Layout(justify_content="center"),
        )

        self.raw_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self.raw_uploader.metadata_table_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        widgets.HBox(
                            [self.raw_uploader.progress_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        self.raw_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )

        self.prenorm_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self.prenorm_uploader.metadata_table_output],
                            layout=widgets.Layout(justify_content="center"),
                        ),
                        self.prenorm_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import Prenormalized Data",),
        )

        self.tab = widgets.VBox(
            [
                self.raw_accordion,
                self.prenorm_accordion,
            ]
        )


class RawUploader_SSRL62B(UploaderBase):
    """
    This uploader has two slots for choosing files and uploading:
    one for references, one for projections. This
    is because our data is stored in two separate folders.
    """

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()

        self.projections = RawProjectionsTiff_SSRL62B()

        self.filedir = pathlib.Path()
        self.filename = pathlib.Path()

        # Save filedir/filename for projections
        self.filedir_projections = pathlib.Path()
        self.filename_projections = pathlib.Path()

        # Save filedir/filename for references
        self.filedir_references = pathlib.Path()
        self.filename_references = pathlib.Path()

        self.user_overwrite_energy = True

        self.Import = Import

        self.filetypes_to_look_for = ["metadata.txt"]
        self.files_not_found_str = "Choose a directory with a metadata.txt file."

        self.projections_found = False
        self.references_found = False

        # Creates the app that goes into the Import object
        self.create_app()

    def _init_widgets(self):

        # File browser for projections
        self.filechooser_projections = FileChooser()
        self.filechooser_projections.register_callback(
            self._update_quicksearch_from_filechooser_projections
        )
        self.filechooser_label_projections = widgets.Label(
            "Raw Projections", style=header_font_style
        )
        self.filechooser_projections.show_only_dirs = True
        self.filechooser_projections.title = "Choose raw projections file directory:"

        # Quick path search textbox
        self.quick_path_search_projections = widgets.Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=widgets.Layout(align_items="stretch"),
        )
        self.quick_path_search_projections.observe(
            self._update_filechooser_from_quicksearch_projections, names="value"
        )

        # File browser for refs
        self.filechooser_references = FileChooser()
        self.filechooser_references.register_callback(
            self._update_quicksearch_from_filechooser_references
        )
        self.filechooser_label_references = widgets.Label(
            "Raw References", style=header_font_style
        )
        self.filechooser_references.show_only_dirs = True
        self.filechooser_references.title = "Choose raw reference file directory:"

        # Quick path search textbox
        self.quick_path_search_references = widgets.Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=widgets.Layout(align_items="stretch"),
        )
        self.quick_path_search_references.observe(
            self._update_filechooser_from_quicksearch_references, names="value"
        )

        self.upload_progress = widgets.IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=widgets.Layout(justify_content="center"),
        )

        # -- Setting metadata widgets --------------------------------------------------

        self.px_size_textbox = widgets.FloatText(
            value=30,
            description="Pixel size (binning 1): ",
            disabled=False,
            style=extend_description_style,
        )
        self.px_units_dropdown_opts = ["nm", "\u00b5m", "mm", "cm"]
        self.px_units_dropdown = widgets.Dropdown(
            value="\u00b5m",
            options=self.px_units_dropdown_opts,
            disabled=False,
            style=extend_description_style,
        )
        self.energy_textbox = widgets.FloatText(
            value=8000,
            description="Energy: ",
            disabled=False,
            style=extend_description_style,
        )
        self.energy_units_dropdown = widgets.Dropdown(
            value="eV",
            options=["eV", "keV"],
            disabled=False,
        )

    def _update_quicksearch_from_filechooser_projections(self, *args):
        self.filedir = pathlib.Path(self.filechooser_projections.selected_path)
        self.filename = self.filechooser_projections.selected_filename
        self.quick_path_search_projections.value = str(self.filedir / self.filename)

    def _update_quicksearch_from_filechooser_references(self, *args):
        self.filedir = pathlib.Path(self.filechooser_references.selected_path)
        self.filename = self.filechooser_references.selected_filename
        self.quick_path_search_references.value = str(self.filedir / self.filename)

    def _update_filechooser_from_quicksearch_projections(self, change):
        self.projections_found = False
        self.looking_in_projections_filedir = True
        self.looking_in_references_filedir = False
        self.import_button.disable()
        self._update_filechooser_from_quicksearch(change)

    def _update_filechooser_from_quicksearch_references(self, change):
        self.references_found = False
        self.looking_in_projections_filedir = False
        self.looking_in_references_filedir = True
        self.import_button.disable()
        self._update_filechooser_from_quicksearch(change)

    def enable_import(self):
        if self.references_found and self.projections_found:
            self.import_button.enable()
            self.projections.import_metadata()
            self.projections.metadata.create_metadata_hbox()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.projections.metadata.metadata_hbox)

    def import_data(self, *args):

        tic = time.perf_counter()
        self.projections.import_data(self)
        toc = time.perf_counter()
        self.projections.metadatas = Metadata.create_metadatas(
            self.projections.metadata.filedir / self.projections.metadata.filename
        )
        self.import_status_label.value = f"Import and normalization took {toc-tic:.0f}s"
        self.projections.filedir = self.projections.import_savedir
        self.viewer.plot(self.projections)

    def update_filechooser_from_quicksearch(self, textfiles):
        try:
            metadata_filepath = (
                self.filedir / [file for file in textfiles if "metadata.txt" in file][0]
            )
        except Exception:
            not_found_str = (
                "This directory doesn't have a metadata.txt file,"
                + " please try another one."
            )
            self.find_metadata_status_label.value = not_found_str
            return
        try:
            assert metadata_filepath != []
        except Exception:
            not_found_str = (
                "This directory doesn't have a metadata.txt file,"
                + " please try another one."
            )
            self.find_metadata_status_label.value = not_found_str
            return
        else:
            if self.looking_in_projections_filedir:
                self.projections_found = True
                self.projections_metadata_filepath = metadata_filepath
                self.projections.import_metadata_projections(self)
                self.filedir_projections = copy.copy(self.filedir)

            if self.looking_in_references_filedir:
                self.references_found = True
                self.references_metadata_filepath = metadata_filepath
                self.projections.import_metadata_references(self)
                self.filedir_references = copy.copy(self.filedir)

            self.enable_import()

    def create_app(self):

        self.app = widgets.HBox(
            [
                widgets.VBox(
                    [
                        self.filechooser_label_projections,
                        widgets.Label("Quick path search:"),
                        widgets.HBox(
                            [
                                self.quick_path_search_projections,
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser_projections,
                        self.filechooser_label_references,
                        widgets.Label("Quick path search:"),
                        self.quick_path_search_references,
                        self.filechooser_references,
                        widgets.HBox(
                            [
                                self.px_size_textbox,
                                self.px_units_dropdown,
                            ]
                        ),
                        widgets.HBox(
                            [
                                self.energy_textbox,
                                self.energy_units_dropdown,
                            ]
                        ),
                        # self.save_tiff_on_import_checkbox,
                    ],
                ),
                self.viewer.app,
            ],
            layout=widgets.Layout(justify_content="center"),
        )


class RawUploader_SSRL62C(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()
        self.user_overwrite_energy = False
        self.projections = RawProjectionsXRM_SSRL62C()
        self.Import = Import
        self.filechooser.title = "Choose a Raw XRM File Directory"
        self.filetypes_to_look_for = [".txt"]
        self.files_not_found_str = "Choose a directory with a ScanInfo file."

        # Creates the app that goes into the Import object
        self.create_app()

    def _init_widgets(self):
        self.upload_progress = widgets.IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=widgets.Layout(justify_content="center"),
        )
        self.energy_select_multiple = widgets.SelectMultiple(
            options=["7700.00", "7800.00", "7900.00"],
            rows=3,
            description="Energies (eV): ",
            disabled=True,
            style=extend_description_style,
        )
        self.energy_select_label = "Select energies"
        self.energy_select_label = widgets.Label(
            self.energy_select_label, style=header_font_style
        )
        self.energy_overwrite_textbox = widgets.FloatText(
            description="Overwrite Energy (eV): ",
            style=extend_description_style,
            disabled=True,
        )
        self.energy_overwrite_textbox.observe(self.energy_overwrite, names="value")

        self.already_uploaded_energies_select = widgets.Select(
            options=["7700.00", "7800.00", "7900.00"],
            rows=3,
            description="Uploaded Energies (eV): ",
            disabled=True,
            style=extend_description_style,
        )
        self.already_uploaded_energies_label = "Previously uploaded energies"
        self.already_uploaded_energies_label = widgets.Label(
            self.already_uploaded_energies_label, style=header_font_style
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

    def import_data(self):

        tic = time.perf_counter()
        self.projections.import_filedir_all(self)
        toc = time.perf_counter()
        self.projections.metadatas = Metadata.create_metadatas(
            self.projections.metadata.filedir / self.projections.metadata.filename
        )
        self.projections.status_label.value = (
            f"Import and normalization took {toc-tic:.0f}s"
        )
        self.projections.filedir = self.projections.import_savedir
        self.viewer.plot(self.projections)

    def update_filechooser_from_quicksearch(self, textfiles):
        try:
            scan_info_filepath = (
                self.filedir / [file for file in textfiles if "ScanInfo" in file][0]
            )
        except Exception:
            not_found_str = (
                "This directory doesn't have a ScanInfo file,"
                + " please try another one."
            )
            self.find_metadata_status_label.value = not_found_str
            return
        try:
            assert scan_info_filepath != []
        except Exception:
            not_found_str = (
                "This directory doesn't have a ScanInfo file,"
                + " please try another one."
            )
            self.find_metadata_status_label.value = not_found_str
            return
        else:
            self.user_overwrite_energy = False
            self.projections.import_metadata(self)
            self.metadata_table = self.projections.metadata.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.projections.metadata.dataframe)
            self.import_button.enable()
            if self.projections.energy_guessed:
                self.energy_overwrite_textbox.disabled = False
                self.energy_overwrite_textbox.value = (
                    self.projections.energies_list_float[0]
                )
            else:
                self.energy_overwrite_textbox.disabled = True
                self.energy_overwrite_textbox.value = 0
            self.check_energy_folders()

    def check_energy_folders(self):
        self.already_uploaded_energies_select.disabled = True
        folders = [pathlib.Path(f) for f in os.scandir(self.filedir) if f.is_dir()]
        reg_exp = re.compile("\d\d\d\d\d.\d\deV")
        ener_folders = map(reg_exp.findall, [str(folder) for folder in folders])
        self.already_uploaded_energies = [
            str(folder[0][:-2]) for folder in ener_folders if (len(folder) > 0)
        ]
        self.already_uploaded_energies_select.options = self.already_uploaded_energies
        self.already_uploaded_energies_select.disabled = False

    def create_app(self):
        self.app = widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.Label("Find data folder", style=header_font_style),
                        widgets.Label("Quick path search:"),
                        widgets.HBox(
                            [
                                self.quick_path_search,
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser,
                        self.energy_select_label,
                        self.energy_select_multiple,
                        self.energy_overwrite_textbox,
                        self.save_tiff_on_import_checkbox,
                        widgets.VBox(
                            [
                                self.already_uploaded_energies_label,
                                self.already_uploaded_energies_select,
                            ],
                            layout=widgets.Layout(align_content="center"),
                        ),
                    ],
                ),
                self.viewer.app,
            ],
            layout=widgets.Layout(justify_content="center"),
        )
