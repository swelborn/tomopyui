import numpy as np

from tomopyui.backend.io import (
    Metadata_Align,
)
from .uploader_base import UploaderBase


class ShiftsUploader(UploaderBase):
    """"""

    def __init__(self, Prep):
        super().__init__()
        self.Prep = Prep
        self.import_button.callback = Prep.add_shift  # callback to add shifts to list
        self.projections = Prep.projections
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
