import time
import logging
import numpy as np
import pathlib
import functools
import re
import os
import json
import tifffile as tf

from ipyfilechooser import FileChooser
from ipyfilechooser.errors import InvalidPathError, InvalidFileNameError
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui._sharedvars import *
from tomopyui.widgets.view import BqImViewer_Import
from tomopyui.backend.io import (
    RawProjectionsHDF5_ALS832,
    RawProjectionsHDF5_APS,
    RawProjectionsXRM_SSRL62C,
    Projections_Prenormalized,
    Metadata_Align,
    Metadata,
    Metadata_ALS_832_Raw,
    Metadata_ALS_832_Prenorm,
    Metadata_APS_Raw,
    Metadata_APS_Prenorm,
    Metadata_General_Prenorm,
)
from tomopyui.widgets import helpers
from tomopyui.widgets.helpers import (
    ReactiveTextButton,
    ReactiveIconButton,
    SwitchOffOnIconButton,
    ImportButton,
)


class ImportBase(ABC):
    """
    An overarching class that controls the rest of the processing pipeline.
    Holds `Uploader` instances, which can be used for uploading data in the form of
    `Projections` instances. The prenorm_uploader is general, in that it can be used for
    any type of data. This is why it is in the base class. The subclasses of ImportBase
    are for creating and holding raw `Uploader` instances.
    """

    def __init__(self):

        # Init raw/prenorm button switches. These are at the top of each of the Prep,
        # Alignment, and Recon tabs to switch between using raw/uploaded data or
        # prenormalized data. See the `ReactiveButton` helper class.
        self.use_raw_button = ReactiveTextButton(
            self.enable_raw,
            "Click to use raw/normalized data from the Import tab.",
            "Updating plots.",
            "Raw/normalized data from Import tab in use for alignment/reconstruction.",
        )
        self.use_prenorm_button = ReactiveTextButton(
            self.disable_raw,
            "Click to use prenormalized data from the Import tab.",
            "Updating plots.",
            "Prenormalized data from Import tab in use for alignment/reconstruction.",
        )

        # Creates the prenormalized uploader (general)
        # raw_uploader is created in the beamline-specific subclasses.
        # Initializes to setting the `Import` instance's projections to be the prenorm
        # projections, but this is switched with the "enable_raw" or "disable_raw"
        # functions and buttons, defined above. Maybe I am getting terminology incorrect
        # but this is kind of like a switchable singleton.
        self.prenorm_uploader = PrenormUploader(self)
        self.projections = self.prenorm_uploader.projections
        self.uploader = self.prenorm_uploader

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

    def disable_raw(self, *args):
        """
        Makes the prenorm_uploader projections the projections used throughout the app.
        Refreshes plots in the other tabs to match these.
        """
        self.use_raw_button.reset_state()
        self.use_raw = False
        self.use_prenorm = True
        self.projections = self.prenorm_uploader.projections
        self.uploader = self.prenorm_uploader
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.Center.refresh_plots()
        self.Prep.refresh_plots()

    def enable_raw(self, *args):
        """
        Makes the raw_uploader projections the projections used throughout the app.
        Refreshes plots in the other tabs to match these.
        """
        self.use_prenorm_button.reset_state()
        self.use_raw = True
        self.use_prenorm = False
        self.projections = self.raw_uploader.projections
        self.uploader = self.raw_uploader
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.Center.refresh_plots()
        self.Prep.refresh_plots()

    @abstractmethod
    def make_tab(self):
        ...


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


class Import_ALS832(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_ALS832(self)
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


class Import_APS(Import_ALS832):
    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_APS(self)
        self.make_tab()


class UploaderBase(ABC):
    """"""

    def __init__(self):
        # Headers style, make it look halfway decent.
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }

        # File browser
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self._update_quicksearch_from_filechooser)
        self.file_chooser_label = Label(
            "Find data folder", style=self.header_font_style
        )
        self.filedir = pathlib.Path()
        self.filename = pathlib.Path()

        # Quick path search textbox
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.quick_path_search.observe(
            self._update_filechooser_from_quicksearch, names="value"
        )
        self.quick_path_label = Label("Quick path search:")

        # Import button, disabled before you put anything into the quick path
        # see helpers class
        self.import_button = ImportButton(self.import_data)

        # Where metadata will be displayed
        self.metadata_table_output = Output()

        # Progress bar showing upload progress
        self.progress_output = Output()

        # Save tiff checkbox
        self.save_tiff_on_import_checkbox = Checkbox(
            description="Save .tif on import.",
            value=False,
            style=extend_description_style,
            disabled=False,
        )

        # Create data visualizer
        self.viewer = BqImViewer_Import()
        self.viewer.create_app()

        # bool for whether or not metadata was imported
        self.imported_metadata = False

        # Will update based on the import status
        self.import_status_label = Label(layout=Layout(justify_content="center"))

        # Will update when searching for metadata
        self.find_metadata_status_label = Label(layout=Layout(justify_content="center"))

    def check_filepath_exists(self, path):
        self.filename = None
        self.filedir = None
        if path.is_dir():
            self.filedir = path
            self.filechooser.reset(path=path)
        elif path.is_file():
            self.filedir = path.parent
            self.filename = str(path.name)
            self.filechooser.reset(path=path.parent, filename=path.name)
        else:
            self.find_metadata_status_label.value = (
                "No file or directory with that name."
            )
            return False
        return True

    def _update_filechooser_from_quicksearch(self, change):
        """
        Checks path to see if it exists, checks file directory for strings in
        self.filetypes_to_look_for. Then it runs the subclass-specific function
        self.update_filechooser_from_quicksearch.

        Parameters
        ----------
        change
            This comes from the callback of the quick search textbox. change.new is
            a str. To inspect what else comes with change.new, you can edit this by
            putting print(change) at the top of this function.
        """
        path = pathlib.Path(change.new)
        self.import_button.disable()
        self.imported_metadata = False
        if not self.check_filepath_exists(path):
            return
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.find_metadata_status_label)
        try:
            found_files = self.projections._file_finder(
                self.filedir, self.filetypes_to_look_for
            )
            assert found_files != []
        except AssertionError:
            filetype_str = [x + " or " for x in self.filetypes_to_look_for[:-1]]
            filetype_str = "".join(filetype_str + [self.filetypes_to_look_for[-1]])
            self.find_metadata_status_label.value = (
                "No "
                + filetype_str
                + " files found in this directory. "
                + self.files_not_found_str
            )
            self.files_found = False
        else:
            # calls subclass method.
            self.files_found = True
            self.update_filechooser_from_quicksearch(found_files)

    def _update_quicksearch_from_filechooser(self):
        """
        Updates the quick search box after selection from the file chooser. This
        triggers self._update_filechooser_from_quicksearch(), so not much logic is
        needed other than setting the filedirectory and filename.
        """
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filedir / self.filename)

    # Each uploader has a method to update the filechooser from the quick search path,
    # and vice versa.
    @abstractmethod
    def update_filechooser_from_quicksearch(self, change):
        ...

    # Each uploader has a method to import data given the filepath chosen in the
    # filechooser/quicksearch box
    @abstractmethod
    def import_data(self):
        ...


class PrenormUploader(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()

        # store parent Import instance for changing "use_raw" or "use_prenorm" buttons
        # after uploading data.
        self.Import = Import
        self.metadatas = None
        self.projections = Projections_Prenormalized()  # see io.py in backend
        self.filechooser.title = "Import prenormalized data:"
        self.viewer.rectangle_selector_on = False  # TODO: Remove?

        # Quick search/filechooser will look for these types of files.
        self.filetypes_to_look_for = [".json", ".npy", ".tif", ".tiff"]
        self.files_not_found_str = ""
        self.filetypes_to_look_for_images = [".npy", ".tif", ".tiff"]

        # Create widgets for required data entry if the prenorm data does not have
        # proper metadata to run with the rest of the program. For ex, these boxes will
        # pop up if trying to import a normalized tiff stack from another program
        self.metadata_input_output = Output()
        self.metadata_input_output_label = Label(
            "Set Metadata Here",
            layout=Layout(justify_content="center"),
            style=self.header_font_style,
        )
        self.start_angle_textbox = FloatText(
            value=-90,
            description="Starting angle (\u00b0): ",
            disabled=True,
            style=extend_description_style,
        )
        self.angle_end_textbox = FloatText(
            value=90,
            description="Ending angle (\u00b0): ",
            disabled=True,
            style=extend_description_style,
        )
        self.px_size_textbox = FloatText(
            value=30,
            description="Pixel size (binning 1): ",
            disabled=True,
            style=extend_description_style,
        )
        self.px_size_units_dropdown_opts = ["nm", "\u00b5m", "mm", "cm"]
        self.px_size_units_dropdown = Dropdown(
            value="nm",
            options=self.px_size_units_dropdown_opts,
            disabled=True,
            style=extend_description_style,
        )
        self.energy_textbox = FloatText(
            value=8000,
            description="Energy: ",
            disabled=True,
            style=extend_description_style,
        )
        self.energy_units_dropdown = Dropdown(
            value="eV",
            options=["eV", "keV"],
            disabled=True,
            style=extend_description_style,
        )
        self.binning_dropdown = Dropdown(
            value=2,
            description="Binning: ",
            options=[("1", 1), ("2", 2), ("4", 4)],
            disabled=True,
            style=extend_description_style,
        )
        self.angular_resolution_textbox = FloatText(
            value=0.25,
            description="Angular Resolution (\u00b0):",
            disabled=True,
            style=extend_description_style,
        )

        # Collection of widgets (list) to enable or disable, depending on whether or not
        # metadata could be imported when choosing a file or file directory
        self.required_parameters = [
            "start_angle",
            "end_angle",
            "pixel_size",
            "pixel_units",
            "energy_float",
            "energy_units",
            "binning",
            "angular_resolution",
        ]
        self.init_required_values = [-90, 90, 30, "nm", 8000, "eV", 2, 0.25]
        self.widgets_to_enable = [
            self.start_angle_textbox,
            self.angle_end_textbox,
            self.px_size_textbox,
            self.px_size_units_dropdown,
            self.energy_textbox,
            self.energy_units_dropdown,
            self.binning_dropdown,
            self.angular_resolution_textbox,
        ]
        self.required_metadata = zip(self.required_parameters, self.widgets_to_enable)

        # Creating callbacks programatically like this required due to namespace
        # issues. Could make this into a metadata widget class in the future. TODO
        for name, widget in self.required_metadata:
            widget.observe(self.create_metadata_callback(name, widget))

        # Selection widget for tifffs and npys in a folder
        self.images_in_dir_select = Select(
            options=[],
            disabled=False,
        )
        self.images_in_dir_select.observe(self.images_in_dir_callback, names="index")

        # If there is many tiffs in the folder, turn this checkbox on
        self.tiff_folder_checkbox = Checkbox(
            description="Tiff Folder?",
            style=extend_description_style,
            value=False,
            disabled=True,
        )

        self.tiff_folder_checkbox.observe(self.tiff_folder_on, names="value")

        self.metadata_widget_box = VBox(
            [
                self.metadata_input_output_label,
                HBox(
                    [
                        self.start_angle_textbox,
                        self.angle_end_textbox,
                        self.angular_resolution_textbox,
                    ]
                ),
                HBox(
                    [
                        self.px_size_textbox,
                        self.px_size_units_dropdown,
                        self.binning_dropdown,
                    ]
                ),
                HBox([self.energy_textbox, self.energy_units_dropdown]),
            ]
        )
        # Creates the app that goes into the Import object
        self.create_app()

    def create_and_display_metadata_tables(self):
        """
        Creates metadata dataframe and displays it in self.metadata_table_output.
        """
        # [
        #     metadata.set_metadata(self.projections)
        #     for metadata in self.projections.metadatas
        # ]
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

    def reset_required_widgets(self):
        """
        Resets metadata widgets to default values. This also sets the metadata dict
        values (widget.value = val triggers metadata setting callbacks)
        """
        for name, val, widget in zip(
            self.required_parameters, self.init_required_values, self.widgets_to_enable
        ):
            if name not in self.projections.metadata.metadata:
                widget.value = val

    def create_metadata_callback(self, name, widget):
        """
        Callback for setting metadata. Creates metadata table and displays it if all
        required metadata are in self.projections.metadata.metadata
        """

        def callback(change):
            if not self.imported_metadata:
                self.projections.metadata.metadata[name] = widget.value
            if all(
                x in self.projections.metadata.metadata
                for x in self.required_parameters
            ):
                self.create_and_display_metadata_tables()

        return callback

    # this was copied from PrenormalizedProjections get_img_shape.
    # TODO: find better place. helper functions?
    def extract_image_sizes(self, image_list):
        size_list = []

        self.tiff_count_in_folder = len(
            [file for file in image_list if file.suffix in [".tiff", ".tif"]]
        )
        for image in image_list:
            if image.suffix == ".tif" or image.suffix == ".tiff":
                with tf.TiffFile(image) as tif:
                    # if you select a file instead of a file path, it will try to
                    # bring in the full filedir
                    if self.tiff_count_in_folder > 1:
                        self.tiff_folder_checkbox.disabled = False
                        self.tiff_folder_checkbox.disabled = False
                    else:
                        self.tiff_folder_checkbox.disabled = True
                        self.tiff_folder_checkbox.value = False
                    try:
                        imagesize = tif.pages[0].tags["ImageDescription"]
                        size = json.loads(imagesize.value)["shape"]
                    except Exception:
                        sizeZ = self.tiff_count_in_folder
                        sizeY = tif.pages[0].tags["ImageLength"].value
                        sizeX = tif.pages[0].tags["ImageWidth"].value
                    else:
                        sizeZ = size[0]
                        sizeY = size[1]
                        sizeX = size[2]

            elif image.suffix == ".npy":
                size = np.load(image, mmap_mode="r").shape
                sizeZ = size[0]
                sizeY = size[1]
                sizeX = size[2]

            size_tuple = (sizeZ, sizeY, sizeX)
            size_list.append(size_tuple)

        return size_list

    def check_for_images(self):
        try:
            self.images_in_dir = self.projections._file_finder_fullpath(
                self.filedir, self.filetypes_to_look_for_images
            )
            assert self.images_in_dir != []
        except AssertionError:
            filetype_str = [x + " or " for x in self.filetypes_to_look_for_images[:-1]]
            filetype_str = "".join(
                filetype_str + [self.filetypes_to_look_for_images[-1]]
            )
            self.find_metadata_status_label.value = (
                "No "
                + filetype_str
                + "files found in this directory. "
                + self.files_not_found_str
            )
            return False
        else:
            self.image_size_list = self.extract_image_sizes(self.images_in_dir)
            self.images_in_dir_select.options = [x.name for x in self.images_in_dir]
            self.images_in_dir_select.index = None
            return True

    def enter_metadata_output(self):
        """
        Enables/disables widgets if they are not/are already in the metadata. Displays
        the box if any of the widgets are not disabled.
        """

        # Zip params/initial values/widgets and set it to default if not in metadata
        self.required_metadata = zip(
            self.required_parameters, self.init_required_values, self.widgets_to_enable
        )
        # if required parameter not in current metadata instance, enable it and set
        # metadata to default value. if it is, disable and set value to the metadata
        # value
        for name, val, widget in self.required_metadata:
            if name not in self.projections.metadata.metadata:
                widget.disabled = False
                widget.value = val
                self.projections.metadata.metadata[name] = val
            else:
                widget.disabled = True
                widget.value = self.projections.metadata.metadata[name]

        # create metadata dataframe. The dataframe will only appear once all the
        # required metadata is inside the metadata instance
        self.create_and_display_metadata_tables()

        # pop up the widget box if any are disabled
        if not all([x.disabled for x in self.widgets_to_enable]):
            with self.metadata_input_output:
                display(self.metadata_widget_box)

    def _update_filechooser_from_quicksearch(self, change):
        self.metadata_input_output.clear_output()
        super()._update_filechooser_from_quicksearch(change)

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
            self.projections.metadatas = Metadata.get_metadata_hierarchy(
                self.metadata_filepath
            )
            self.metadata_already_displayed = False
            if self.projections.metadatas != []:
                [
                    metadata.set_attributes_from_metadata(self.projections)
                    for metadata in self.projections.metadatas
                ]
                self.create_and_display_metadata_tables()
                self.metadata_already_displayed = True
                self.imported_metadata = True
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

    def import_data(self):
        """
        Function that calls on io.py (projections) to run import. The function chosen
        will depend on whether one is uploading a folder, or a
        """

        with self.metadata_table_output:
            # self.metadata_table_output.clear_output(wait=True)
            # if self.imported_metadata:
            #     [display(m) for m in self.dataframes if m is not None]
            # else:
            #     self.create_and_display_metadata_tables()
            display(self.import_status_label)
        if self.filename == "" or self.filename is None:
            self.projections.import_filedir_projections(self)
        else:
            self.projections.import_file_projections(self)
        self.import_status_label.value = (
            "Plotting data (downsampled for viewer to 0.25x)."
        )
        self.viewer.plot(self.projections)
        self.Import.use_raw_button.reset_state()
        self.Import.use_prenorm_button.reset_state()
        if "import_time" in self.projections.metadata.metadata:
            self.import_status_label.value = (
                "Import, downsampling (if any), and"
                + " plotting complete in "
                + f"~{self.projections.metadata.metadata['import_time']:.0f}s."
            )

    def create_app(self):
        self.app = HBox(
            [
                VBox(
                    [
                        self.quick_path_label,
                        HBox(
                            [
                                self.quick_path_search,
                                VBox(
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
                        self.metadata_input_output,
                    ],
                ),
                self.viewer.app,
            ],
            layout=Layout(justify_content="center"),
        )


class TwoEnergyUploader(PrenormUploader):
    """"""

    def __init__(self, viewer):
        UploaderBase.__init__(self)
        self.projections = Projections_Prenormalized()
        self.filechooser.title = "Import prenormalized data:"
        self.viewer = viewer
        self.viewer.create_app()
        self.imported_metadata = False
        self.viewer.rectangle_selector_on = False
        self.energy_textbox = FloatText(
            description="Energy: ",
            disabled=True,
            style=extend_description_style,
        )
        self.pixel_size_textbox = FloatText(
            description="Pixel Size: ",
            disabled=True,
            style=extend_description_style,
        )
        self.widgets_to_enable = [self.energy_textbox, self.pixel_size_textbox]

    def import_data(self):
        tic = time.perf_counter()
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            if self.imported_metadata:
                display(self.projections.metadata.dataframe)
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
        if not self.imported_metadata:
            self.projections.energy = self.energy_textbox.value
            self.projections.current_pixel_size = self.pixel_size_textbox.value
        self.viewer.plot(self.projections)
        toc = time.perf_counter()


class ShiftsUploader(UploaderBase):
    """"""

    def __init__(self, Prep):
        super().__init__()
        self.Prep = Prep
        self.import_button.callback = Prep.add_shift  # callback to add shifts to list
        self.projections = Prep.imported_projections
        self.filechooser.title = "Import shifts: "
        self.imported_metadata = False
        self.filetypes_to_look_for = ["sx.npy", "sy.npy", "alignment_metadata.json"]
        self.files_not_found_str = ""

    def update_filechooser_from_quicksearch(self, shifts_files):
        self.import_button.disable()
        self.shifts_from_json = False
        self.shifts_from_npy = False
        if "sx.npy" in shifts_files:  # TODO: eventually deprecate
            self.shifts_from_json = False
            self.shifts_from_npy = True
        elif "alignment_metadata.json" in shifts_files:
            self.shifts_from_json = True
            self.shifts_from_npy = False
        if self.shifts_from_json:
            self.align_metadata_filepath = self.filedir / "alignment_metadata.json"
            self.imported_metadata = False
            self.import_shifts_from_metadata()
            self.update_shift_lists()
            self.imported_metadata = True
        else:
            self.imported_metadata = False
            self.import_shifts_from_npy()
            self.update_shift_lists()

    def import_shifts_from_npy(self):
        self.sx = np.load(self.filedir / "sx.npy")
        self.sy = np.load(self.filedir / "sy.npy")
        self.conv = np.load(self.filedir / "conv.npy")
        self.align_metadata = Metadata_Align()
        self.align_metadata.filedir = self.filedir
        self.align_metadata.filename = "alignment_metadata.json"
        self.align_metadata.filepath = (
            self.align_metadata.filedir / "alignment_metadata.json"
        )
        self.align_metadata.load_metadata()

    def import_shifts_from_metadata(self):
        self.align_metadata = Metadata_Align()
        self.align_metadata.filedir = self.filedir
        self.align_metadata.filename = "alignment_metadata.json"
        self.align_metadata.filepath = (
            self.align_metadata.filedir / "alignment_metadata.json"
        )
        self.align_metadata.load_metadata()
        self.sx = self.align_metadata.metadata["sx"]
        self.sy = self.align_metadata.metadata["sy"]
        self.conv = self.align_metadata.metadata["convergence"]

    def update_shift_lists(self):
        self.Prep.shifts_sx_select.options = self.sx
        self.Prep.shifts_sy_select.options = self.sy
        self.import_button.enable()

    def import_data(self, change):
        pass


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


class RawUploader_ALS832(UploaderBase):
    """
    Raw uploaders are the way you get your raw data (projections, flats, dark fields)
    into TomoPyUI. It holds a ProjectionsBase subclass (see io.py) that will do all of
    the data import stuff. the ProjectionsBase subclass for SSRL is
    RawProjectionsXRM_SSRL62. For you, it could be named
    RawProjectionsHDF5_APSyourbeamlinenumber().

    """

    def __init__(self, Import):
        super().__init__()  # look at UploaderBase __init__()
        self._init_widgets()
        self.projections = RawProjectionsHDF5_ALS832()
        self.reset_metadata_to = Metadata_ALS_832_Raw
        self.Import = Import
        self.filechooser.title = "Import Raw hdf5 File"
        self.filetypes_to_look_for = [".h5"]
        self.files_not_found_str = "Choose a directory with an hdf5 file."

        # Creates the app that goes into the Import object
        self.create_app()

    def _init_widgets(self):
        """
        You can make your widgets more fancy with this function. See the example in
        RawUploader_SSRL62C.
        """
        pass

    def import_data(self):
        """
        This is what is called when you click the blue import button on the frontend.
        """
        with self.progress_output:
            self.progress_output.clear_output()
            display(self.import_status_label)
        tic = time.perf_counter()
        self.projections.import_file_all(self)
        toc = time.perf_counter()
        self.import_status_label.value = f"Import and normalization took {toc-tic:.0f}s"
        self.viewer.plot(self.projections)

    def update_filechooser_from_quicksearch(self, h5files):
        """
        This is what is called when you update the quick path search bar. Right now,
        this is very basic. If you want to see a more complex version of this you can
        look at the example in PrenormUploader.

        This is called after _update_filechooser_from_quicksearch in UploaderBase.
        """
        if len(h5files) == 1:
            self.filename = h5files[0]
        elif len(h5files) > 1 and self.filename is None:
            self.find_metadata_status_label.value = (
                "Multiple h5 files found in this"
                + " directory. Choose one with the file browser."
            )
            self.import_button.disable()
            return
        self.projections.metadata = self.reset_metadata_to()
        self.projections.import_metadata(self.filedir / self.filename)
        self.projections.metadata.metadata_to_DataFrame()
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.projections.metadata.dataframe)
            self.import_button.enable()

    def create_app(self):
        self.app = HBox(
            [
                VBox(
                    [
                        self.quick_path_label,
                        HBox(
                            [
                                self.quick_path_search,
                                self.import_button.button,
                            ]
                        ),
                        self.filechooser,
                    ],
                ),
                self.viewer.app,
            ],
            layout=Layout(justify_content="center"),
        )


class RawUploader_APS(RawUploader_ALS832):
    """
    See descriptions in RawUploader_ALS832 superclass. You shouldn't have to do much
    here other than changing self.projections and self.reset_metadata_to if you change
    those names.
    # Francesco: edit here, if needed.
    """

    def __init__(self, Import):
        super().__init__(Import)
        self.projections = RawProjectionsHDF5_APS()
        self.reset_metadata_to = Metadata_APS_Raw
