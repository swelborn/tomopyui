import time

from ipywidgets import *

from tomopyui.widgets.styles import (
    extend_description_style,
)
from tomopyui.backend.io import (
    Projections_Prenormalized,
)
from .prenorm import PrenormUploader
from .uploader_base import UploaderBase
from IPython.display import display
from .uploader_base import UploaderBase


class TwoEnergyUploader(PrenormUploader):
    """"""

    def __init__(self, viewer):
        UploaderBase.__init__(self)
        # Quick search/filechooser will look for these types of files.
        self.filetypes_to_look_for = [".json", ".npy", ".tif", ".tiff", ".hdf5", ".h5"]
        self.files_not_found_str = ""
        self.filetypes_to_look_for_images = [".npy", ".tif", ".tiff", ".hdf5", ".h5"]
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

    def _update_filechooser_from_quicksearch(self, change):
        UploaderBase._update_filechooser_from_quicksearch(self, change)

    def check_for_images(self):
        return False

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
        self.import_status_label.value = "Plotting data."
        if not self.imported_metadata:
            self.projections.energy = self.energy_textbox.value
            self.projections.current_pixel_size = self.pixel_size_textbox.value
        self.viewer.plot(self.projections)
        toc = time.perf_counter()
