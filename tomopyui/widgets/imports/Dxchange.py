import time

import ipywidgets as widgets

from tomopyui.backend.io import (
    Metadata,
    Metadata_ALS_832_Raw,
    RawProjectionsHDF5_ALS832,
)
from IPython.display import display
from .importer_base import ImportBase
from .uploader_base import UploaderBase


class Import_Dxchange(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.raw_uploader = RawUploader_Dxchange(self)
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


class RawUploader_Dxchange(UploaderBase):
    """
    Raw uploaders are the way you get your raw data (projections, flats, dark fields)
    into TomoPyUI. It holds a ProjectionsBase subclass (see io.py) that will do all of
    the data import stuff. the ProjectionsBase subclass for SSRL is
    RawProjectionsXRM_SSRL62.
    """
    filetypes_to_look_for = [".h5", ".hdf5"]
    def __init__(self, Import):
        super().__init__()  # look at UploaderBase __init__()
        self._init_widgets()
        self.projections = RawProjectionsHDF5_ALS832()
        self.reset_metadata_to = Metadata_ALS_832_Raw
        self.Import = Import
        self.filechooser.title = "Import Raw hdf5 File"
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
        self.projections.metadatas = Metadata.create_metadatas(
            self.projections.metadata.filedir / self.projections.metadata.filename
        )
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
        self.app = widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.Label("Quick path search:"),
                        widgets.HBox(
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
            layout=widgets.Layout(justify_content="center"),
        )
