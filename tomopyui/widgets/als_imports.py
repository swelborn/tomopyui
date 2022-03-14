import numpy as np
import pathlib
import os
import dxchange
import time
import pandas as pd
import shutil
from tomopyui.backend.io import (
    RawProjectionsBase,
    Metadata_ALS_832_Raw,
    Metadata_ALS_832_Prenorm,
)
from ipywidgets import *
from tomopyui._sharedvars import *
from tomopyui.widgets.view import BqImViewer_Import
from tomopyui.widgets.imports import ImportBase, UploaderBase


class Import_ALS832(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.raw_projections = RawProjectionsHDF5_ALS832()
        self.raw_uploader = RawUploader_ALS832(self)
        self.use_raw_button.on_click(self.enable_raw)
        self.use_prenorm_button.on_click(self.disable_raw)
        self.make_tab()

    def make_tab(self):

        self.switch_data_buttons = HBox(
            [self.use_raw_button, self.use_prenorm_button],
            layout=Layout(justify_content="center"),
        )

        raw_import = HBox(
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
                    ],
                ),
                self.raw_uploader.viewer.app,
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
                        raw_import,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )
        norm_import = HBox(
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
                    ],
                ),
                self.prenorm_uploader.viewer.app,
            ],
            layout=Layout(justify_content="center"),
        )

        self.prenorm_accordion = Accordion(
            children=[norm_import],
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
                self.switch_data_buttons,
                self.raw_accordion,
                self.prenorm_accordion,
                self.meta_accordion,
            ]
        )


class RawProjectionsHDF5_ALS832(RawProjectionsBase):
    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".h5"]
        self.angles_from_filenames = True
        self.metadata = Metadata_ALS_832_Raw()

    def import_metadata(self, filepath):
        self.metadata.filedir = filepath.parent
        self.metadata.filename = filepath.name
        self.metadata.filepath = filepath
        self.pxY = int(
            dxchange.read_hdf5(
                filepath, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.pxX = int(
            dxchange.read_hdf5(
                filepath, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.px_size = (
            dxchange.read_hdf5(filepath, "/measurement/instrument/detector/pixel_size")[
                0
            ]
            / 10.0
        )  # /10 to convert units from mm to cm
        self.pxZ = int(
            dxchange.read_hdf5(filepath, "/process/acquisition/rotation/num_angles")[0]
        )
        self.propagation_dist = dxchange.read_hdf5(
            filepath, "/measurement/instrument/camera_motor_stack/setup/camera_distance"
        )[1]
        self.energy = (
            dxchange.read_hdf5(
                filepath, "/measurement/instrument/monochromator/energy"
            )[0]
            / 1000
        )
        self.angular_range = dxchange.read_hdf5(
            filepath, "/process/acquisition/rotation/range"
        )[0]

        self.metadata.set_metadata(self)
        self.metadata.filename = "raw_metadata.json"

    def import_filedir_all(self, filedir):
        pass

    def import_filedir_projections(self, filedir):
        pass

    def import_filedir_flats(self, filedir):
        pass

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, filepath):
        (
            self._data,
            self.flats,
            self.darks,
            self.angles_rad,
        ) = dxchange.exchange.read_aps_tomoscan_hdf5(filepath)
        self.norm_filedir = self.filedir / str(filepath.stem)
        if os.path.exists(self.norm_filedir):
            shutil.rmtree(self.norm_filedir)
        os.makedirs(self.norm_filedir)
        self.data = self._data
        self.angles_deg = (180 / np.pi) * self.angles_rad
        self.metadata.set_metadata(self)
        self.metadata.save_metadata()
        self.imported = True

    def import_file_projections(self, filepath):
        tomo_grp = "/".join([exchange_base, "data"])
        tomo = dxreader.read_hdf5(fname, tomo_grp, slc=(proj, sino), dtype=dtype)

    def import_file_flats(self, filepath):
        flat_grp = "/".join([exchange_base, "data_white"])
        flat = dxreader.read_hdf5(fname, flat_grp, slc=(None, sino), dtype=dtype)

    def import_file_darks(self, filepath):
        dark_grp = "/".join([exchange_base, "data_dark"])
        dark = dxreader.read_hdf5(fname, dark_grp, slc=(None, sino), dtype=dtype)

    def import_file_angles(self, filepath):
        theta_grp = "/".join([exchange_base, "theta"])
        theta = dxreader.read_hdf5(fname, theta_grp, slc=None)


class RawUploader_ALS832(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()
        self.projections = Import.raw_projections
        self.Import = Import
        self.import_button.on_click(self.import_data)
        self.projections.filedir = self.filedir
        self.projections.filename = self.filename
        self.viewer = BqImViewer_Import()
        self.viewer.create_app()
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Raw hdf5 file"

    def _init_widgets(self):
        self.metadata_table_output = Output()
        self.progress_output = Output()

    def import_data(self, change):
        self.projections.filedir = self.filedir
        self.projections.filename = self.filename
        self.projections.filepath = self.filedir / self.filename
        self.projections.import_metadata(self.projections.filepath)
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        with self.progress_output:
            display(
                Label(
                    "Importing data",
                    layout=Layout(justify_content="center"),
                )
            )
        self.projections.import_file_all(self.projections.filepath)
        with self.progress_output:
            display(Label("Normalizing", layout=Layout(justify_content="center")))
        self.projections.normalize()
        with self.progress_output:
            display(
                Label(
                    "Saving projections as npy for faster IO",
                    layout=Layout(justify_content="center"),
                )
            )
        _metadata = self.projections.metadata.metadata.copy()
        self.projections.filedir = self.projections.norm_filedir
        self.projections.save_normalized_as_npy()
        self.projections._check_downsampled_data()
        self.projections.metadata = Metadata_ALS_832_Prenorm()
        self.projections.metadata.filedir = self.projections.filedir
        self.projections.metadata.filename = "import_metadata.json"
        self.projections.metadata.metadata = _metadata
        self.projections.metadata.metadata["metadata_type"] = "ALS832_Normalized"
        self.projections.metadata.metadata["data_hierarachy_level"] = 1
        toc = time.perf_counter()
        self.projections.metadata.metadata["import_time"] = toc - tic
        self.projections.metadata.set_metadata()
        self.projections.metadata.save_metadata()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"
        with self.progress_output:
            display(
                Label(
                    f"Import and normalization took {toc-tic:.0f}s",
                    layout=Layout(justify_content="center"),
                )
            )
        self.viewer.plot(self.projections)

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filedir = path
        self.filechooser.reset(path=path)

    def update_quicksearch_from_filechooser(self):
        self.filedir = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filedir)
        # metadata must be set here in case tomodata is created (for filedir
        # import). this can be changed later.

        self.projections.import_metadata(self.filedir / self.filename)
        self.projections.metadata.metadata_to_DataFrame()

        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.projections.metadata.dataframe)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
        # except:
        # self.filename = None
        # with self.metadata_table_output:
        #     self.metadata_table_output.clear_output(wait=True)
        #     print(
        #         "Selected file either isn't .h5 or does not have correct metadata, please try another one."
        #     )
