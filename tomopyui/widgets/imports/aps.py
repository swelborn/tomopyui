
from tomopyui.widgets.imports.als import Import_ALS832, RawUploader_ALS832
from tomopyui.widgets.view import BqImViewer_Projections_Parent
from tomopyui.backend.io import (
    Metadata_APS_Raw,
    Metadata_APS_Prenorm,
)


class Import_APS(Import_ALS832):
    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_APS(self)
        self.make_tab()




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
