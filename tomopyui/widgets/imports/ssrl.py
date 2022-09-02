

from tomopyui.widgets.imports.imports import ImportBase, UploaderBase, Projections_Prenormalized
from tomopyui.backend.io.metadata import Metadata
from tomopyui.backend.io.ssrl import (
    RawProjectionsXRM_SSRL62C,
    RawProjectionsTiff_SSRL62B,
)
from ipywidgets import *
from ipyfilechooser import FileChooser
from tomopyui._sharedvars import *
import pathlib

class Import_SSRL62B(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.angles_from_filenames = True
        self.raw_uploader = RawUploader_SSRL62B(self)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = HBox(
            [self.use_raw_button.button, self.use_prenorm_button.button],
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
                        self.raw_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )

        self.prenorm_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.prenorm_uploader.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        self.prenorm_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import Prenormalized Data",),
        )

        self.tab = VBox(
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
        self.filechooser_label_projections = Label(
            "Raw Projections", style=self.header_font_style
        )
        self.filechooser_projections.show_only_dirs = True
        self.filechooser_projections.title = "Choose raw projections file directory:"

        # Quick path search textbox
        self.quick_path_search_projections = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_search_projections.observe(
            self._update_filechooser_from_quicksearch_projections, names="value"
        )
        self.quick_path_label_projections = Label("Quick path search (projections):")

        # File browser for refs
        self.filechooser_references = FileChooser()
        self.filechooser_references.register_callback(
            self._update_quicksearch_from_filechooser_references
        )
        self.filechooser_label_references = Label(
            "Raw References", style=self.header_font_style
        )
        self.filechooser_references.show_only_dirs = True
        self.filechooser_references.title = "Choose raw reference file directory:"

        # Quick path search textbox
        self.quick_path_search_references = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_search_references.observe(
            self._update_filechooser_from_quicksearch_references, names="value"
        )
        self.quick_path_label_references = Label("Quick path search (references):")

        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )

        # -- Setting metadata widgets --------------------------------------------------

        self.px_size_textbox = FloatText(
            value=30,
            description="Pixel size (binning 1): ",
            disabled=False,
            style=extend_description_style,
        )
        self.px_units_dropdown_opts = ["nm", "\u00b5m", "mm", "cm"]
        self.px_units_dropdown = Dropdown(
            value="\u00b5m",
            options=self.px_units_dropdown_opts,
            disabled=False,
            style=extend_description_style,
        )
        self.energy_textbox = FloatText(
            value=8000,
            description="Energy: ",
            disabled=False,
            style=extend_description_style,
        )
        self.energy_units_dropdown = Dropdown(
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
        self.projections.metadatas = Metadata.get_metadata_hierarchy(
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
            # self.projections.import_metadata(self)
            # self.metadata_table = self.projections.metadata.metadata_to_DataFrame()
            # with self.metadata_table_output:
            #     self.metadata_table_output.clear_output(wait=True)
            #     display(self.projections.metadata.dataframe)
            
    def create_app(self):

        self.app = HBox(
            [
                VBox(
                    [
                        self.filechooser_label_projections,
                        self.quick_path_label_projections,
                        HBox(
                            [
                                self.quick_path_search_projections,
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser_projections,
                        self.filechooser_label_references,
                        self.quick_path_label_references,
                        self.quick_path_search_references,
                        self.filechooser_references,
                        HBox(
                            [
                                self.px_size_textbox,
                                self.px_units_dropdown,
                            ]
                        ),
                        HBox(
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
            layout=Layout(justify_content="center"),
        )


class Import_SSRL62C(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.angles_from_filenames = True
        self.raw_uploader = RawUploader_SSRL62C(self)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = HBox(
            [self.use_raw_button.button, self.use_prenorm_button.button],
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
                        self.raw_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )

        self.prenorm_accordion = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.prenorm_uploader.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        self.prenorm_uploader.app,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import Prenormalized Data",),
        )

        self.tab = VBox(
            [
                # self.switch_data_buttons,
                self.raw_accordion,
                self.prenorm_accordion,
            ]
        )

    def create_app(self):

        self.app = HBox(
            [
                VBox(
                    [
                        self.filechooser_label_projections,
                        self.quick_path_label_projections,
                        HBox(
                            [
                                self.quick_path_search_projections,
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser_projections,
                        self.filechooser_label_references,
                        self.quick_path_label_references,
                        self.quick_path_search_references,
                        self.filechooser_references,
                        HBox(
                            [
                                self.px_size_textbox,
                                self.px_units_dropdown,
                            ]
                        ),
                        HBox(
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
            layout=Layout(justify_content="center"),
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
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
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

    def import_data(self):

        tic = time.perf_counter()
        self.projections.import_filedir_all(self)
        toc = time.perf_counter()
        self.projections.metadatas = Metadata.get_metadata_hierarchy(
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
        self.app = HBox(
            [
                VBox(
                    [
                        self.file_chooser_label,
                        self.quick_path_label,
                        HBox(
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
                        VBox(
                            [
                                self.already_uploaded_energies_label,
                                self.already_uploaded_energies_select,
                            ],
                            layout=Layout(align_content="center"),
                        ),
                    ],
                ),
                self.viewer.app,
            ],
            layout=Layout(justify_content="center"),
        )