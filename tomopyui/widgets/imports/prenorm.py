import copy

import ipywidgets as widgets
from tomopyui.backend.helpers import extract_image_sizes, file_finder_fullpath

from tomopyui.widgets.styles import (
    extend_description_style,
)
from tomopyui.backend.io import (
    Metadata,
    Metadata_General_Prenorm,
    Projections_Prenormalized,
)
from tomopyui.widgets.imports.importer_base import ImportBase
from IPython.display import display
from .uploader_base import UploaderBase
from typing import Optional
from .metadata_input import MetadataInput, WidgetGroupingConfig
from tomopyui.backend.schemas.prenorm_input import (
    PrenormProjectionsMetadata,
    groupings as prenorm_groupings,
)


class PrenormUploader(UploaderBase):
    """"""

    def __init__(self, imp: ImportBase):
        super().__init__()
        self.imp: ImportBase = imp
        self.metadatas: Optional[list[Metadata]] = None
        self.projections: Projections_Prenormalized = Projections_Prenormalized()
        self.filechooser.title = "Import prenormalized data:"
        self.filetypes_to_look_for = [".json", ".npy", ".tif", ".tiff", ".hdf5", ".h5"]
        self.filetypes_to_look_for_images = [".npy", ".tif", ".tiff", ".hdf5", ".h5"]
        self.metadata_input = MetadataInput(
            PrenormProjectionsMetadata, WidgetGroupingConfig(groups=prenorm_groupings)
        )
        self.metadata_table: Optional[widgets.Widget] = None

        # Quick search/filechooser will look for these types of files.
        self.files_not_found_str = ""

        # Selection widget for tifffs and npys in a folder
        self.images_in_dir_select = widgets.Select(options=[], disabled=False)
        self.images_in_dir_select.observe(self.images_in_dir_callback, names="index")

        # If there is many tiffs in the folder, turn this checkbox on
        self.tiff_folder_checkbox = widgets.Checkbox(
            description="Tiff Folder?",
            style=extend_description_style,
            value=False,
            disabled=True,
        )

        self.tiff_folder_checkbox.observe(self.tiff_folder_on, names="value")
        # Creates the app that goes into the Import object
        self.create_app()

    def create_and_display_metadata_tables(self):
        """
        Creates metadata dataframe and displays it in self.metadata_table_output.
        """
        self.projections.update_metadatas()
        [metadata.create_metadata_box() for metadata in self.projections.metadatas]
        self.metadata_vboxes = [x.metadata_vbox for x in self.projections.metadatas]
        if not self.metadata_already_displayed:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                [display(m) for m in self.metadata_vboxes]

    def tiff_folder_on(self, change):
        """
        Turns on the tiff folder option, where it will try to load all the tiffs in the
        folder (a tiff sequence). Some redundancies
        """
        if change.new:
            self.projections.metadata.metadata["tiff_folder"] = True
            self.projections.tiff_folder = True
            self.tiff_folder = True
            self.projections.metadata.metadata["pxZ"] = self.tiff_count_in_folder
            self.projections.metadata.metadata["pxX"] = self.image_size_list[0][2]
            self.projections.metadata.metadata["pxY"] = self.image_size_list[0][1]
            self.import_button.enable()
            self.images_in_dir_select.disabled = True
            self.create_and_display_metadata_tables()
        if not change.new:
            if self.images_in_dir_select.index is not None:
                self.import_button.enable()
            else:
                self.import_button.disable()
            self.images_in_dir_select.disabled = False
            self.projections.metadata.metadata["tiff_folder"] = False
            self.projections.tiff_folder = False
            self.tiff_folder = False
            self.images_in_dir_callback(None, from_select=False)

    def images_in_dir_callback(self, change, from_select=True):
        """
        Callback for the image selection widget. Displays updated metadata table with
        image pixel sizes if you select a tiff or npy.

        Parameters
        ----------
        from_select: bool
            If this is false, ind will be the current selected index. Needed for
            calling back from self.tiff_folder_on.
        """
        if not from_select:
            ind = self.images_in_dir_select.index
        else:
            ind = change.new
        if ind is not None:
            self.projections.metadata.metadata["pxX"] = self.image_size_list[ind][2]
            self.projections.metadata.metadata["pxY"] = self.image_size_list[ind][1]
            self.projections.metadata.metadata["pxZ"] = self.image_size_list[ind][0]
            self.projections.filedir = self.filedir
            self.filename = str(self.images_in_dir[ind].name)
            self.projections.filename = str(self.images_in_dir[ind].name)
            self.create_and_display_metadata_tables()
            self.import_button.enable()
        else:
            self.import_button.disable()

    def check_for_images(self) -> bool:
        """
        Checks for images in the specified directory and updates the UI components
        based on the findings. It uses the extract_image_sizes function to determine
        the size of images found and update the tiff_folder_checkbox accordingly.

        Returns:
            bool: True if images are found, False otherwise.
        """
        try:
            # Attempt to find images in the directory
            self.images_in_dir = file_finder_fullpath(
                self.filedir, self.filetypes_to_look_for_images
            )
            if not self.images_in_dir:
                raise AssertionError

            # Extract the sizes of found images and update UI components
            self.image_size_list, tiff_count = extract_image_sizes(
                self.images_in_dir, self.tiff_folder_checkbox, self.projections
            )

            # Update options for the select widget based on found images
            self.images_in_dir_select.options = [x.name for x in self.images_in_dir]
            self.images_in_dir_select.index = None

        except AssertionError:
            # Handle case where no images are found
            filetype_str = " or ".join(self.filetypes_to_look_for_images)
            self.find_metadata_status_label.value = f"No {filetype_str} files found in this directory. {self.files_not_found_str}"
            return False

        return True

    def _update_filechooser_from_quicksearch(self, change):
        self.metadata_input.clear_output()
        super()._update_filechooser_from_quicksearch(change)

    def update_filechooser_from_quicksearch_new(self, json_files):
        self.images_in_dir = None
        metadata_files = [file for file in json_files if "_metadata" in file]

        # Check for metadata files and set up projections metadata accordingly
        if not metadata_files:
            # No metadata files found
            self.handle_no_metadata_found()
        else:
            # Metadata files found, process them
            self.process_metadata_files(metadata_files[0])

        # After handling metadata, check for images
        images_found = self.check_for_images()
        self.update_status_and_ui_based_on_images_found(images_found)

    def handle_no_metadata_found(self):
        self.imported_metadata = False
        self.metadata_already_displayed = False
        self.find_metadata_status_label.value = (
            "No metadata associated with this file. "
            "Please enter metadata below before uploading "
            "so that tomopyui functions properly."
        )
        self.projections.metadata = Metadata_General_Prenorm()
        self.projections.metadata.filedir = self.filedir
        self.projections.metadatas = [self.projections.metadata]
        self.enter_metadata_output()

    def process_metadata_files(self, metadata_file):
        self.metadata_filepath = self.filedir / metadata_file
        self.projections.metadatas = Metadata.create_metadatas(self.metadata_filepath)
        self.metadata_already_displayed = False
        if self.projections.metadatas:
            self.setup_projections_metadata()
            self.import_button.enable()
        else:
            self.find_metadata_status_label.value = (
                "This directory has metadata but no prenormalized data you can upload."
            )

    def setup_projections_metadata(self):
        parent = {}
        for i, metadata in enumerate(self.projections.metadatas):
            metadata.filepath = copy.copy(self.metadata_filepath)
            if i == 0:
                metadata.load_metadata()
            else:
                metadata.metadata = parent
            metadata.set_attributes_from_metadata(self.projections)
            if "parent_metadata" in metadata.metadata:
                parent = metadata.metadata["parent_metadata"].copy()

        self.create_and_display_metadata_tables()
        self.metadata_already_displayed = True
        self.imported_metadata = True
        self.determine_active_metadata()

    def determine_active_metadata(self):
        if (
            len(self.projections.metadatas) > 1
            and self.projections.metadatas[-1].metadata["metadata_type"]
            == "General_Normalized"
        ):
            self.projections.metadata = self.projections.metadatas[-1]
        else:
            self.projections.metadata = self.projections.metadatas[0]

    def update_status_and_ui_based_on_images_found(self, images_found):
        if images_found:
            self.metadata_input.clear_output()
            self.update_find_metadata_status_label_based_on_filename()
        else:
            self.find_metadata_status_label.value = (
                "This directory has metadata but no prenormalized data you can upload."
            )

    def update_find_metadata_status_label_based_on_filename(self):
        if self.filename:
            self.find_metadata_status_label.value = (
                "This directory has all the metadata you need. "
                "If your images are a lot of separate images, then upload the directory now. "
                "Otherwise, select a single image to upload using the file browser."
            )
        else:
            self.find_metadata_status_label.value = (
                "This directory has all the metadata you need. "
                "Proceed to upload your data (click blue button)."
            )

    def import_data(self):
        """
        Function that calls on io.py (projections) to run import. The function chosen
        will depend on whether one is uploading a folder, or a
        """

        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            if self.imported_metadata:
                self.metadata_input_output.clear_output()
                self.create_and_display_metadata_tables()
            else:
                self.metadata_input_output.clear_output()

            display(self.import_status_label)
        if (
            self.filename == ""
            or self.filename is None
            and self.imported_metadata
            and not self.tiff_folder
        ):
            self.projections.import_filedir_projections(self)
        else:
            self.projections.import_file_projections(self)
        self.import_status_label.value = (
            "Plotting data (downsampled for viewer to 0.5x)."
        )
        self.viewer.plot(self.projections, ds=True, no_check=True)
        self.imp.use_raw_button.reset_state()
        self.imp.use_prenorm_button.reset_state()
        if "import_time" in self.projections.metadata.metadata:
            self.import_status_label.value = (
                "Import, downsampling (if any), and"
                + " plotting complete in "
                + f"~{self.projections.metadata.metadata['import_time']:.0f}s."
            )
        self.imp.use_prenorm_button.run_callback()

    def create_app(self):
        self.app = widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.Label("Quick path search:"),
                        widgets.HBox(
                            [
                                self.quick_path_search,
                                widgets.VBox(
                                    [
                                        self.images_in_dir_select,
                                        self.tiff_folder_checkbox,
                                        self.save_tiff_on_import_checkbox,
                                    ]
                                ),
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser,
                        self.metadata_input,
                    ],
                ),
                self.viewer.app,
            ],
            layout=widgets.Layout(justify_content="center"),
        )

    def update_filechooser_from_quicksearch(self, json_files):
        self.images_in_dir = None
        try:
            self.metadata_filepath = [
                self.filedir / file for file in json_files if "_metadata" in file
            ]
            assert self.metadata_filepath != []
        except AssertionError:  # this means no metadata files in this directory
            self.imported_metadata = False
            self.metadata_already_displayed = False
            if self.check_for_images():

                # Initialize new metadata - old one might not have correct values
                self.projections.metadata = Metadata_General_Prenorm()
                self.projections.metadata.filedir = self.filedir
                self.projections.metadatas = [self.projections.metadata]
                self.enter_metadata_output()
                self.find_metadata_status_label.value = (
                    "No metadata associated with this file. "
                    + "Please enter metadata below before uploading "
                    + "so that tomopyui functions properly."
                )
            else:
                self.find_metadata_status_label.value = (
                    "This directory has no metadata"
                    + " files and no images that you can upload."
                )
                self.import_button.disable()
        else:
            self.metadata_filepath = self.metadata_filepath[0]
            self.projections.metadatas = Metadata.create_metadatas(
                self.metadata_filepath
            )
            self.metadata_already_displayed = False
            if self.projections.metadatas != []:
                parent = {}
                for i, metadata in enumerate(self.projections.metadatas):
                    metadata.filepath = copy.copy(self.metadata_filepath)
                    if i == 0:
                        metadata.load_metadata()
                    else:
                        metadata.metadata = parent
                    metadata.set_attributes_from_metadata(self.projections)
                    if "parent_metadata" in metadata.metadata:
                        parent = metadata.metadata["parent_metadata"].copy()
                self.create_and_display_metadata_tables()
                self.metadata_already_displayed = True
                self.imported_metadata = True
                self.import_button.enable()
                if len(self.projections.metadatas) > 1:
                    if (
                        self.projections.metadatas[-1].metadata["metadata_type"]
                        == "General_Normalized"
                    ):
                        self.projections.metadata = self.projections.metadatas[-1]
                    else:
                        self.projections.metadata = self.projections.metadatas[-2]
                else:
                    self.projections.metadata = self.projections.metadatas[0]
                if self.check_for_images():
                    self.metadata_input_output.clear_output()
                    if self.filename is not None:
                        self.find_metadata_status_label.value = (
                            "This directory has all the metadata you need. "
                            + " Proceed to upload your data (click blue button)."
                        )
                    else:
                        self.find_metadata_status_label.value = (
                            "This directory has all the metadata you need."
                            + " If your images are a lot of separate"
                            + " images, then upload the directory now. Otherwise,"
                            + " select a single image to upload using the file browser."
                        )
                else:
                    self.find_metadata_status_label.value = (
                        "This directory has metadata"
                        + " but no prenormalized data you can upload."
                    )
