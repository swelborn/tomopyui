from abc import ABC, abstractmethod
import numpy as np
import pathlib
import tifffile as tf
import tomopy.prep.normalize as tomopy_normalize
import os
import json
import dxchange
import re
from tomopy.sim.project import angles as angle_maker
import olefile
import pandas as pd
from tomopyui.backend.io import IOBase, ProjectionsBase, RawProjectionsBase

from tomopyui.widgets import helpers
from ipyfilechooser import FileChooser
import logging
from ipywidgets import *
from tomopyui._sharedvars import *
import time
from tomopyui.widgets.view import BqImViewer_Import
from tomopyui.widgets.imports import ImportBase, UploaderBase


class Import_ALS832(ImportBase):
    """"""

    def __init__(self):
        super().__init__()
        self.angles_from_filenames = True
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
                self.prenorm_uploader.plotter.app,
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

    def import_metadata(self, filename):

        self.metadata["numslices"] = int(
            dxchange.read_hdf5(
                filename, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.metadata["numrays"] = int(
            dxchange.read_hdf5(
                filename, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.metadata["pxsize"] = (
            dxchange.read_hdf5(filename, "/measurement/instrument/detector/pixel_size")[
                0
            ]
            / 10.0
        )  # /10 to convert units from mm to cm
        self.metadata["num_angles"] = int(
            dxchange.read_hdf5(filename, "/process/acquisition/rotation/num_angles")[0]
        )
        self.metadata["propagation_dist"] = dxchange.read_hdf5(
            filename, "/measurement/instrument/camera_motor_stack/setup/camera_distance"
        )[1]
        self.metadata["kev"] = (
            dxchange.read_hdf5(
                filename, "/measurement/instrument/monochromator/energy"
            )[0]
            / 1000
        )
        self.metadata["angularrange"] = dxchange.read_hdf5(
            filename, "/process/acquisition/rotation/range"
        )[0]

        self.pxZ = self.metadata["num_angles"]
        self.pxY = self.metadata["numslices"]
        self.pxX = self.metadata["numrays"]
        self.energy = self.metadata["kev"]

    def import_filedir_all(self, filedir):
        pass

    def import_filedir_projections(self, filedir):
        pass

    def import_filedir_flats(self, filedir):
        pass

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, fullpath):
        self.import_metadata(fullpath)

        (
            self._data,
            self.flats,
            self.darks,
            self.angles_rad,
        ) = dxchange.exchange.read_aps_tomoscan_hdf5(fullpath)

        self.data = self._data
        self.angles_deg = (180 / np.pi) * self.angles_rad
        self.imported = True

    def import_file_projections(self, fullpath):
        tomo_grp = "/".join([exchange_base, "data"])
        tomo = dxreader.read_hdf5(fname, tomo_grp, slc=(proj, sino), dtype=dtype)

    def import_file_flats(self, fullpath):
        flat_grp = "/".join([exchange_base, "data_white"])
        flat = dxreader.read_hdf5(fname, flat_grp, slc=(None, sino), dtype=dtype)

    def import_file_darks(self, fullpath):
        dark_grp = "/".join([exchange_base, "data_dark"])
        dark = dxreader.read_hdf5(fname, dark_grp, slc=(None, sino), dtype=dtype)

    def import_file_angles(self, fullpath):
        theta_grp = "/".join([exchange_base, "theta"])
        theta = dxreader.read_hdf5(fname, theta_grp, slc=None)

    def set_options_from_frontend(self, Import, Uploader):
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        self.angles_from_filenames = Import.angles_from_filenames
        self.upload_progress = Uploader.upload_progress

    def metadata_to_DataFrame(self):

        # create headers and data for table
        top_headers = []
        middle_headers = []
        data = []
        # Image information
        top_headers.append(["Image Information"])
        middle_headers.append(["X Pixels", "Y Pixels", "Num. θ"])
        data.append([self.pxX, self.pxY, self.pxZ])

        top_headers.append(["Experiment Settings"])
        middle_headers.append(
            ["Energy (keV)", "Propagation Distance (mm)", "Angular range (deg)"]
        )
        data.append(
            [
                self.energy,
                self.metadata["propagation_dist"],
                self.metadata["angularrange"],
            ]
        )

        # create dataframe with the above settings
        df = pd.DataFrame(
            [data[0]],
            columns=pd.MultiIndex.from_product([top_headers[0], middle_headers[0]]),
        )
        for i in range(len(middle_headers)):
            if i == 0:
                continue
            else:
                newdf = pd.DataFrame(
                    [data[i]],
                    columns=pd.MultiIndex.from_product(
                        [top_headers[i], middle_headers[i]]
                    ),
                )
                df = df.join(newdf)

        # set datatable styles
        s = df.style.hide(axis="index")
        s.set_table_styles(
            {
                ("Acquisition Information", "Repeat Scan"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Image Information", "X Pixels"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Mosaic Information", "Up"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            [
                {"selector": "th.col_heading", "props": "text-align: center;"},
                {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
                {"selector": "td", "props": "text-align: center;" "font-size: 1.2em;"},
                {
                    "selector": "th:not(.index_name)",
                    "props": "background-color: #0F52BA; color: white;",
                },
            ],
            overwrite=False,
        )

        return s

    def string_num_totuple(self, s):
        return (
            "".join(c for c in s if c.isalpha()) or None,
            "".join(c for c in s if c.isdigit() or None),
        )


class RawUploader_ALS832(UploaderBase):
    """"""

    def __init__(self, Import):
        super().__init__()
        self._init_widgets()
        self.projections = Import.raw_projections
        self.Import = Import
        self.import_button.on_click(self.import_data)
        self.projections.set_options_from_frontend(self.Import, self)
        self.plotter = BqImViewer_Import()
        self.plotter.create_app()
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Raw hdf5 file"

    def _init_widgets(self):
        self.metadata_table_output = Output()
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )

    def import_data(self, change):

        self.projections.set_options_from_frontend(self.Import, self)
        # self.projections.import_file_projections(os.path.join(self.filedir, self.filepath))

        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        self.upload_progress.value = 0
        self.upload_progress.max = self.projections.pxZ
        with self.progress_output:
            display(self.upload_progress)

        self.projections.import_file_all(os.path.join(self.filedir, self.filename))
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
        self.projections.save_normalized_as_npy()
        toc = time.perf_counter()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"
        with self.progress_output:
            display(
                Label(
                    f"Import and normalization took {toc-tic:.0f}s",
                    layout=Layout(justify_content="center"),
                )
            )
        self.plotter.plot(self.projections.prj_imgs, self.filedir)

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

        try:
            self.projections.import_metadata(os.path.join(self.filedir, self.filename))
            self.metadata_table = self.projections.metadata_to_DataFrame()

            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
                self.import_button.button_style = "info"
                self.import_button.disabled = False
        except:
            self.filename = None
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "Selected file either isn't .h5 or does not have correct metadata, please try another one."
                )
