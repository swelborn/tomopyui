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
from tomopyui.backend.util.dxchange.reader import read_ole_metadata, read_xrm
import olefile
import pandas as pd
from skimage.transform import rescale
import pathlib
from joblib import Parallel, delayed
from ipywidgets import *
import tempfile
import dask.array as da
import dask
import os
import shutil
import copy
import multiprocessing as mp
from functools import partial


class IOBase:
    def __init__(self):

        self._data = np.random.rand(10, 100, 100)
        self.imported = False
        self._fullpath = None
        self.dtype = None
        self.shape = None
        self.pxX = self._data.shape[2]
        self.pxY = self._data.shape[1]
        self.pxZ = self._data.shape[0]
        self.size_gb = None
        self.filedir = None
        self.filename = None
        self.extension = None
        self.parent = None
        self.energy = None
        self.raw = False
        self.single_file = False
        self.metadata = {}

    @property
    def data(self):
        return self._data

    @data.setter
    # set data info whenever setting data
    def data(self, value):
        (self.pxZ, self.pxY, self.pxX) = self._data.shape
        self.rangeX = (0, self.pxX - 1)
        self.rangeY = (0, self.pxY - 1)
        self.rangeZ = (0, self.pxZ - 1)
        self.size_gb = self._data.nbytes / 1048576 / 1000
        self.dtype = self._data.dtype
        self._data = value

    @property
    def fullpath(self):
        return self._fullpath

    @fullpath.setter
    def fullpath(self, value):
        self.filedir = value.parent
        self.filename = value.name
        self.extension = value.suffix
        self._fullpath = value

    def _check_downsampled_data(self, energy=None, label=None):
        if energy is not None:
            filedir = self.energy_filedir
        else:
            filedir = self.filedir
        try:
            self.filedir_ds = pathlib.Path(filedir / "downsampled").mkdir(parents=True)
            self.filedir_ds = pathlib.Path(filedir / "downsampled")
        except FileExistsError:
            self.filedir_ds = pathlib.Path(filedir / "downsampled")
            try:
                if label is not None:
                    label.value = "Loading premade downsampled data and histograms."
                self._load_ds_and_hists()
            except Exception:
                if label is not None:
                    label.value = (
                        "Downsampled folder exists, but doesn't match"
                        + "format. Writing new downsampled data."
                    )
                self._write_downsampled_data()
        else:
            if label is not None:
                label.value = "Writing new downsampled data."
            self._write_downsampled_data()

    def _write_downsampled_data(self):
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in self.ds_vals]
        #TODO: make parallel on individual slices of data. see bottom of this .py (archived code)
        # ds_data = [rescale_parallel_pool(self.pxZ, self.data, ds_vals_list) for ds_vals_list in self.ds_vals]
        # :
        #     ds_data.append(rescale_parallel_pool(self.pxZ, self.data, self.ds_vals))
        ds_data = Parallel(n_jobs=int(os.environ["num_cpu_cores"]))(
            delayed(rescale)(self.data, x) for x in self.ds_vals
        )
        ds_data.append(self.data)
        ds_data_da = [da.from_array(x) for x in ds_data]
        ranges = [[np.min(x), np.max(x)] for x in ds_data]
        hists = [
            da.histogram(x, range=[y[0], y[1]], bins=200)
            for x, y in zip(ds_data_da, ranges)
        ]
        hist_intensities = [hist[0] for hist in hists]
        hist_intensities = [hist.compute() for hist in hist_intensities]
        bin_edges = [hist[1] for hist in hists]
        xvals = [[(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)] for b in bin_edges]
        for data, string in zip(ds_data, ds_vals_strs):
            np.save(self.filedir_ds / str("ds" + string), data)
        for hist_int, string, bin_edge, bin_center in zip(
            hist_intensities, ds_vals_strs, bin_edges, xvals
        ):
            np.savez(
                self.filedir_ds / str("ds" + string + "hist"),
                frequency=hist_int,
                edges=bin_edge,
                bin_centers=bin_center,
            )
        self._load_ds_and_hists()


    def _load_ds_and_hists(self):
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in self.ds_vals]
        self.hists = [
            np.load(self.filedir_ds / str("ds" + string + "hist.npz"))
            for string in ds_vals_strs
        ]
        self.data_ds = [
            np.load(self.filedir_ds / str("ds" + string + ".npy"), mmap_mode="r")
            for string in ds_vals_strs
        ]

    def _file_finder(self, filedir, filetypes: list):
        files = [pathlib.PurePath(f) for f in os.scandir(filedir) if not f.is_dir()]
        files_with_ext = [
            file.name for file in files if any(x in file.name for x in filetypes)
        ]
        return files_with_ext


class ProjectionsBase(IOBase, ABC):
    # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
    aliases = {
        "prj_imgs": "data",
        "num_angles": "pxZ",
        "width": "pxX",
        "height": "pxY",
        "pixel_range_x": "rangeX",
        "pixel_range_y": "rangeY",
        "pixel_range_z": "rangeZ",  # Could probably fix these
    }

    def __init__(self):
        super().__init__()
        self.ds_vals = [(1, 0.25, 0.25)]
        self.angles_rad = None
        self.angles_deg = None
        self.allowed_extensions = [".npy", ".tiff", ".tif"]

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    def save_normalized_as_npy(self, energy=None):
        if energy is not None:
            np.save(self.energy_filedir / str("normalized_projections.npy"), self.data)
        else:
            np.save(self.filedir / str("normalized_projections.npy"), self.data)

    def save_normalized_as_tiff(self, energy=None):
        if energy is not None:
            tf.imwrite(
                self.energy_filedir / str("normalized_projections.tif"),
                self.data,
            )
        else:
            tf.imwrite(self.filedir / str("normalized_projections.tif"), self.data)

    @abstractmethod
    def import_metadata(self, filedir):
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
        ...

    @abstractmethod
    def import_file_projections(self, fullpath):
        ...


class Projections_Prenormalized(ProjectionsBase, ABC):
    def import_filedir_projections(self, Uploader):
        self.filedir = Uploader.filedir
        if Uploader.imported_metadata:
            self.fullpath = Uploader.filedir / "normalized_projections.npy"
            self._data = np.load(
                Uploader.filedir / "normalized_projections.npy"
            ).astype(np.float32)
            self.set_attributes_from_metadata()
        else:
            self.angle_start = Uploader.angle_start
            self.angle_end = Uploader.angle_end
            cwd = os.getcwd()
            os.chdir(Uploader.filedir)
            image_sequence = tf.TiffSequence()
            self._data = image_sequence.asarray().astype(np.float32)
            self.data = self._data
            image_sequence.close()
            self.make_angles()
            os.chdir(cwd)

    def import_file_projections(self, Uploader):
        self.imported = False
        self.filedir = Uploader.filedir
        self._fullpath = Uploader.filedir / Uploader.filename
        self.fullpath = self._fullpath
        if Uploader.imported_metadata:
            Uploader.import_status_label.value = (
                "Detected import_metadata.json in this directory,"
                + " uploading normalized_projections.npy."
            )
            self._data = np.load(
                Uploader.filedir / "normalized_projections.npy"
            ).astype(np.float32)
            self.set_attributes_from_metadata()
            self.imported = True

        elif ".tif" in str(self.fullpath):
            # if there is a file name, checks to see if there are many more
            # tiffs in the filedir. If there are, will upload all of them.
            filetypes = [".tif", ".tiff"]
            tifffiles = self._file_finder(self.fullpath.parent, filetypes)
            if len(tifffiles) > 50:
                Uploader.import_status_label.value = (
                    "Detected tiff stack in this directory, uploading all of them. "
                    + "Using angles from textboxes below."
                )
                self.import_filedir_projections(self.fullpath.parent)
                return
            (
                "Did not detect imported metadata, uploading single .tif file from this"
                + " folder. Using angles from textboxes below."
            )
            self._data = np.array(
                dxchange.reader.read_tiff(self.fullpath).astype(np.float32)
            )
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self.data = self._data
            self.angle_start = Uploader.angle_start
            self.angle_end = Uploader.angle_end
            self.make_angles()
            self.imported = True

        elif ".npy" in str(self.fullpath):
            Uploader.import_status_label.value = (
                "Did not detect imported metadata. Uploading .npy. "
                + "Using angles from textboxes below."
            )
            self._data = np.load(self.fullpath).astype(np.float32)
            self._fullpath = self.fullpath
            self.fullpath = self._fullpath
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self.data = self._data
            self.angle_start = Uploader.angle_start
            self.angle_end = Uploader.angle_end
            self.make_angles()
            self._check_downsampled_data(self.energy)
            self.imported = True

    def get_img_shape(self):

        if self.extension == ".tif" or self.extension == ".tiff":
            allowed_extensions = [".tiff", ".tif"]
            file_list = [
                pathlib.PurePath(f) for f in os.scandir(self.filedir) if not f.is_dir()
            ]
            tiff_file_list = [
                file.name
                for file in file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            tiff_count_in_filedir = len(tiff_file_list)
            with tf.TiffFile(self.fullpath) as tif:
                # if you select a file instead of a file path, it will try to
                # bring in the full filedir
                if tiff_count_in_filedir > 50:
                    sizeX = tif.pages[0].tags["ImageWidth"].value
                    sizeY = tif.pages[0].tags["ImageLength"].value
                    sizeZ = tiff_count_in_filedir  # can maybe use this later
                else:
                    imagesize = tif.pages[0].tags["ImageDescription"]
                    size = json.loads(imagesize.value)["shape"]
                    sizeZ = size[0]
                    sizeY = size[1]
                    sizeX = size[2]

        elif self.extension == ".npy":
            size = np.load(self.fullpath, mmap_mode="r").shape
            sizeZ = size[0]
            sizeY = size[1]
            sizeX = size[2]

        return (sizeZ, sizeY, sizeX)

    def set_prj_ranges(self):
        self.pixel_range_x = (0, self.prj_shape[2] - 1)
        self.pixel_range_y = (0, self.prj_shape[1] - 1)
        self.pixel_range_z = (0, self.prj_shape[0] - 1)

    def make_angles(self):
        self.angles_rad = angle_maker(
            self.pxZ,
            ang1=self.angle_start,
            ang2=self.angle_end,
        )
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    @abstractmethod
    def import_metadata(self):
        ...

    @abstractmethod
    def metadata_to_DataFrame(self):
        ...

    @abstractmethod
    def set_attributes_from_metadata(self):
        ...


class Projections_Prenormalized_SSRL62(Projections_Prenormalized):
    def import_metadata(self, Uploader):
        self.metadata = load_metadata(fullpath=Uploader.import_metadata_filepath)
        self.scan_info = self.metadata["scan_info"]

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        exp_time_proj = f"{self.metadata['projections_exposure_time']:0.2f}"
        exp_time_ref = f"{self.metadata['references_exposure_time']:0.2f}"
        ds_vals = self.metadata["downsampled_values"]
        ds_vals = [x[2] for x in ds_vals]
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                "Scan Type": self.metadata["scan_type"],
                "Ref. Exp. Time": exp_time_ref,
                "Proj. Exp. Time": exp_time_proj,
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. θ": self.metadata["num_angles"],
                "Binning": self.metadata["binning"],
            },
            {
                "Energy Overwritten": self.metadata["user_overwrite_energy"],
                ".tif Saved": self.metadata["saved_as_tiff"],
                "Downsample Values": ds_vals,
            },
        ]
        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])
        top_headers.append(["Other Information"])

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
                ("Acquisition Information", middle_headers[0][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Other Information", middle_headers[2][0]): [
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

    def set_attributes_from_metadata(self):
        self.scan_info = copy.deepcopy(self.metadata["scan_info"])
        self.scan_info["FILES"] = [
            pathlib.Path(file) for file in self.metadata["scan_info"]["FILES"]
        ]
        self.scan_info_path = self.metadata["scan_info_path"]
        self.run_script_path = pathlib.Path(self.metadata["run_script_path"])
        self.flats_filenames = [
            pathlib.Path(file) for file in self.metadata["flats_filenames"]
        ]
        self.data_filenames = [
            pathlib.Path(file) for file in self.metadata["projections_filenames"]
        ]
        self.scan_type = self.metadata["scan_type"]
        self.scan_order = self.metadata["scan_order"]
        self.pxX = self.metadata["pxX"]
        self.pxY = self.metadata["pxY"]
        self.pxZ = self.metadata["pxZ"]
        self.angles_rad = self.metadata["angles_rad"]
        self.angles_deg = self.metadata["angles_deg"]
        self.start_angle = self.metadata["start_angle"]
        self.end_angle = self.metadata["end_angle"]
        self.binning = self.metadata["binning"]
        self.energies_list_float = self.metadata["all_raw_energies_float"]
        self.energies_list_str = self.metadata["all_raw_energies_str"]
        self.user_overwrite_energy = self.metadata["user_overwrite_energy"]
        self.current_energy_str = self.metadata["energy_str"]
        self.current_energy_float = self.metadata["energy_float"]
        self.energy_units = self.metadata["energy_units"]
        self.raw_pixel_sizes = self.metadata["all_raw_pixel_sizes"]
        self.pixel_size_from_metadata = self.metadata["pixel_size_from_scan_info"]
        self.current_pixel_size = self.metadata["pixel_size"]
        self.pixel_units = self.metadata["pixel_units"]
        self.size_gb = self.metadata["normalized_projections_size_gb"]
        self.energy_filedir = pathlib.Path(
            self.metadata["normalized_projections_directory"]
        )
        self.filedir_ds = pathlib.Path(
            self.metadata["downsampled_projections_directory"]
        )
        self.ds_vals = self.metadata["downsampled_values"]
        self.saved_as_tiff = self.metadata["saved_as_tiff"]


class RawProjectionsBase(ProjectionsBase, ABC):
    def __init__(self):
        super().__init__()
        self.flats = None
        self.flats_ind = None
        self.darks = None
        self.normalized = False

    def normalize_nf(self):
        self._data = tomopy_normalize.normalize_nf(
            self._data, self.flats, self.darks, self.flats_ind
        )
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    # for Tomopy-based normalization
    def normalize(self):
        self._data = tomopy_normalize.normalize(self._data, self.flats, self.darks)
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    # For dask-based normalization (should be faster)
    def average_chunks(chunked_da):
        @dask.delayed
        def mean_on_chunks(a):
            return np.mean(a, axis=0)[np.newaxis, ...]

        blocks = chunked_da.to_delayed().ravel()
        results = [
            da.from_delayed(
                mean_on_chunks(b),
                shape=(1, chunked_da.shape[1], chunked_da.shape[2]),
                dtype=np.float32,
            )
            for b in blocks
        ]
        # arr not computed yet
        arr = da.concatenate(results, axis=0, allow_unknown_chunksizes=True)
        return arr

    def normalize_and_average(
        projs, flats, dark, flat_loc, num_exposures_per_proj, status_label=None
    ):
        """
        Function takes pre-chunked dask arrays of projections, flats, darks, along
        with flat locations within the projection images.

        The chunk size is the number of exposures per reference or per angle.

        projs should look like this:
        Proj1
        Proj1
        Proj2
        Proj2
        ...
        Proj10
        Proj10

        flats should look like this:
        flat1
        flat1
        flat1
        flat2
        flat2
        flat2

        darks darks does not need to be location-specific, it can just be an ndarray
        (only the median will be used across whole dark dataset)

        Normalization procedure will do the following:
        1. Average flats. Array above will turn into this:
        flat1average
        flat2average
        2. Subtract darks from average flats
        3. Subtract darks from individual projections (not averaged)
        4. Chunk the projections into bulks that are close to flats
        5. Normalize each projection (not averaged) based on the proximity to the
        nearest flat
        6. Average the projections in their initial chunks

        """
        if status_label is not None:
            status_label.value = "Averaging flatfields."
        # Averaging flats
        flats_reduced = RawProjectionsBase.average_chunks(flats)
        dark = np.median(dark, axis=0)
        denominator = flats_reduced - dark
        denominator = denominator.compute()
        # Projection locations defined as the centerpoint between two reference
        # collections
        # Chunk the projections such that they will be divided by the nearest flat
        # The first chunk of data will be divided by the first flat.
        # The first chunk of data is likely smaller than the others.
        proj_locations = [
            int(np.ceil((flat_loc[i] + flat_loc[i + 1]) / 2))
            for i in range(len(flat_loc) - 1)
        ]
        chunk_setup = [int(np.ceil(proj_locations[0]))]
        for i in range(len(proj_locations) - 1):
            chunk_setup.append(proj_locations[i + 1] - proj_locations[i])
        chunk_setup.append(projs.shape[0] - sum(chunk_setup))
        chunk_setup = tuple(chunk_setup)
        projs_rechunked = projs.rechunk({0: chunk_setup, 1: -1, 2: -1})  # chunk data
        projs_rechunked = projs_rechunked - dark
        if status_label is not None:
            status_label.value = f"Dividing by flatfields and taking -log."

        # Don't know if putting @classmethod above a decorator will mess it up, so this
        # fct is inside. This is kind of funky. TODO.

        @dask.delayed
        def divide_arrays(x, ind):
            y = denominator[ind]
            return np.true_divide(x, y)

        blocks = projs_rechunked.to_delayed().ravel()
        results = [
            da.from_delayed(
                divide_arrays(b, i),
                shape=(chunksize, projs_rechunked.shape[1], projs_rechunked.shape[2]),
                dtype=np.float32,
            )
            for i, (b, chunksize) in enumerate(zip(blocks, chunk_setup))
        ]
        arr = da.concatenate(results, axis=0, allow_unknown_chunksizes=True)
        arr = arr.rechunk((num_exposures_per_proj, -1, -1))
        arr = RawProjectionsBase.average_chunks(arr).astype(np.float32)
        arr = -da.log(arr)
        arr = arr.compute()
        return arr

    @abstractmethod
    def import_metadata(self, filedir):
        ...

    @abstractmethod
    def import_filedir_all(self, filedir):
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
        ...

    @abstractmethod
    def import_filedir_flats(self, filedir):
        ...

    @abstractmethod
    def import_filedir_darks(self, filedir):
        ...

    @abstractmethod
    def import_file_all(self, fullpath):
        ...

    @abstractmethod
    def import_file_projections(self, fullpath):
        ...

    @abstractmethod
    def import_file_flats(self, fullpath):
        ...

    @abstractmethod
    def import_file_darks(self, fullpath):
        ...


class RawProjectionsXRM_SSRL62(RawProjectionsBase):
    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True

    def import_metadata(self, Uploader):
        filetypes = [".txt"]
        textfiles = self._file_finder(Uploader.filedir, filetypes)
        self.scan_info_path = [
            Uploader.filedir / file for file in textfiles if "ScanInfo" in file
        ][0]
        self.parse_scan_info()
        self.determine_scan_type()
        self.run_script_path = [
            Uploader.filedir / file for file in textfiles if "ScanInfo" not in file
        ]
        if len(self.run_script_path) == 1:
            self.run_script_path = self.run_script_path[0]
        elif len(self.run_script_path) > 1:
            for file in self.run_script_path:
                with open(file, "r") as f:
                    line = f.readline()
                    if line.startswith(";;"):
                        self.run_script_path = file
        (
            self.flats_filenames,
            self.data_filenames,
        ) = self.get_all_data_filenames()
        # assume that the first projection is the same as the rest for metadata
        self.scan_info["PROJECTION_METADATA"] = self.read_xrms_metadata(
            [self.data_filenames[0]]
        )
        if self.angles_from_filenames:
            self.get_angles_from_filenames()
        else:
            self.get_angles_from_metadata()
        self.pxZ = len(self.angles_rad)
        self.pxY = self.scan_info["PROJECTION_METADATA"][0]["image_height"]
        self.pxX = self.scan_info["PROJECTION_METADATA"][0]["image_width"]
        self.binning = self.scan_info["PROJECTION_METADATA"][0]["camera_binning"]
        self.raw_data_type = self.scan_info["PROJECTION_METADATA"][0]["data_type"]
        if self.raw_data_type == 5:
            self.raw_data_type = np.dtype(np.uint16)
        elif self.raw_data_type == 10:
            self.raw_data_type = np.dtype(np.float32)
        self.pixel_size_from_metadata = (
            self.scan_info["PROJECTION_METADATA"][0]["pixel_size"] * 1000
        )  # nm
        self.get_and_set_energies(Uploader)
        self.filedir = Uploader.filedir

    def import_filedir_all(self, Uploader):
        self.user_overwrite_energy = Uploader.user_overwrite_energy
        self.filedir = Uploader.filedir
        self.selected_energies = Uploader.energy_select_multiple.value
        if len(self.selected_energies) == 0:
            self.selected_energies = (Uploader.energy_select_multiple.options[0],)
            Uploader.energy_select_multiple.value = (
                Uploader.energy_select_multiple.options[0],
            )
        self.import_from_run_script(Uploader)
        self.imported = True

    def import_filedir_projections(self, filedir):
        pass

    def import_filedir_flats(self, filedir):
        pass

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, fullpath):
        pass

    def import_file_projections(self, fullpath):
        pass

    def import_file_flats(self, fullpath):
        pass

    def import_file_darks(self, fullpath):
        pass

    def set_metadata(self):
        self.metadata["scan_info"] = copy.deepcopy(self.scan_info)
        self.metadata["scan_info"]["FILES"] = [
            str(file) for file in self.metadata["scan_info"]["FILES"]
        ]
        self.metadata["scan_info_path"] = str(self.scan_info_path)
        self.metadata["run_script_path"] = str(self.run_script_path)
        self.metadata["flats_filenames"] = [str(file) for file in self.flats_filenames]
        self.metadata["projections_filenames"] = [
            str(file) for file in self.data_filenames
        ]
        self.metadata["scan_type"] = self.scan_type
        self.metadata["scan_order"] = self.scan_order
        self.metadata["pxX"] = self.pxX
        self.metadata["pxY"] = self.pxY
        self.metadata["pxZ"] = self.pxZ
        self.metadata["num_angles"] = self.pxZ
        self.metadata["angles_rad"] = self.angles_rad
        self.metadata["angles_deg"] = self.angles_deg
        self.metadata["start_angle"] = float(self.angles_deg[0])
        self.metadata["end_angle"] = float(self.angles_deg[-1])
        self.metadata["binning"] = self.binning
        self.metadata["projections_exposure_time"] = self.metadata["scan_info"][
            "PROJECTION_METADATA"
        ][0]["exposure_time"]
        self.metadata["references_exposure_time"] = self.metadata["scan_info"][
            "FLAT_METADATA"
        ][0]["exposure_time"]
        self.metadata["all_raw_energies_float"] = self.energies_list_float
        self.metadata["all_raw_energies_str"] = self.energies_list_str
        self.metadata["user_overwrite_energy"] = self.user_overwrite_energy
        self.metadata["energy_str"] = self.current_energy_str
        self.metadata["energy_float"] = self.current_energy_float
        self.metadata["energy_units"] = "eV"
        self.metadata["all_raw_pixel_sizes"] = self.raw_pixel_sizes
        self.metadata["pixel_size_from_scan_info"] = self.pixel_size_from_metadata
        self.metadata["pixel_size"] = self.current_pixel_size
        self.metadata["pixel_units"] = "nm"
        self.metadata["raw_projections_dtype"] = str(self.raw_data_type)
        self.metadata["raw_projections_directory"] = str(self.data_filenames[0].parent)
        self.metadata["normalized_projections_dtype"] = str(np.dtype(np.float32))
        self.metadata["normalized_projections_size_gb"] = self.size_gb
        self.metadata["normalized_projections_directory"] = str(self.energy_filedir)
        self.metadata["normalized_projections_filename"] = "normalized_projections.npy"
        self.metadata["normalization_function"] = "dask"
        self.metadata["downsampled_projections_directory"] = str(self.filedir_ds)
        self.metadata["downsampled_values"] = self.ds_vals
        self.metadata["saved_as_tiff"] = self.saved_as_tiff

    def set_options_from_frontend(self, Import, Uploader):
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        self.angles_from_filenames = Import.angles_from_filenames

    def parse_scan_info(self):
        data_file_list = []
        self.scan_info = []
        with open(self.scan_info_path, "r") as f:
            filecond = True
            for line in f.readlines():
                if "FILES" not in line and filecond:
                    self.scan_info.append(line.strip())
                    filecond = True
                else:
                    filecond = False
                    _ = self.scan_info_path.parent / line.strip()
                    data_file_list.append(_)
        metadata_tp = map(self.string_num_totuple, self.scan_info)
        self.scan_info = {scanvar[0]: scanvar[1] for scanvar in metadata_tp}
        self.scan_info["REFEVERYEXPOSURES"] = self.scan_info["REFEVERYEXPOSURES"][1:]
        self.scan_info = {key: int(self.scan_info[key]) for key in self.scan_info}
        self.scan_info["FILES"] = data_file_list[1:]

    def determine_scan_type(self):
        self.scan_order = [
            (k, self.scan_info[k])
            for k in ("TOMO", "ENERGY", "MOSAIC", "MULTIEXPOSURE")
            if self.scan_info[k] != 0
        ]
        self.scan_order = sorted(self.scan_order, key=lambda x: x[1])
        self.scan_type = [string for string, val in self.scan_order]
        self.scan_type = "_".join(self.scan_type)

    def get_and_set_energies(self, Uploader):
        self.energy_guessed = False
        energies = []
        with open(self.run_script_path, "r") as f:
            for line in f.readlines():
                if line.startswith("sete "):
                    energies.append(float(line[5:]))
        self.energies_list_float = sorted(list(set(energies)))
        if self.energies_list_float == []:
            self.energies_list_float = [
                self.est_en_from_px_size(self.pixel_size_from_metadata, self.binning)
            ]
            self.energy_guessed = True
        self.energies_list_str = [
            f"{energy:08.2f}" for energy in self.energies_list_float
        ]
        self.raw_pixel_sizes = [
            self.calculate_px_size(energy, self.binning)
            for energy in self.energies_list_float
        ]
        Uploader.energy_select_multiple.options = self.energies_list_str
        Uploader.energy_select_multiple.value = [self.energies_list_str[0]]
        if len(self.energies_list_str) > 10:
            Uploader.energy_select_multiple.rows = 10
        else:
            Uploader.energy_select_multiple.rows = len(self.energies_list_str)
        if len(self.energies_list_str) == 1 and self.energy_guessed:
            Uploader.energy_select_multiple.disabled = True
            Uploader.energy_select_multiple.description = "Est. Energy (eV):"
            Uploader.energy_overwrite_textbox.disabled = False
        else:
            Uploader.energy_select_multiple.description = "Energies (eV):"
            Uploader.energy_select_multiple.disabled = False
            Uploader.energy_overwrite_textbox.disabled = True

    def calculate_px_size(self, energy, binning):
        # From Johanna's calibration doc
        pixel_size = 0.002039449 * energy - 0.792164997
        pixel_size = pixel_size * binning
        return pixel_size

    def est_en_from_px_size(self, pixel_size, binning):
        # From Johanna's calibration doc
        energy = (pixel_size / binning + 0.792164997) / 0.002039449
        return energy

    # def import_from_metadata(self):
    #     for file in self.scan_info["FILES"]:
    #         for line in f.readlines():
    #             if line.startswith("sete "):
    #                 energies.append(float(line[5:]))
    #                 flats.append([])
    #                 collects.append([])
    #             elif line.startswith("collect "):
    #                 filename = line[8:].strip()
    #                 if "ref_" in filename:
    #                     flats[-1].append(self.run_script_path.parent / filename)
    #                 else:
    #                     collects[-1].append(self.run_script_path.parent / filename)
    #     energies.pop(0)
    #     flats.pop(0)
    #     collects.pop(0)
    #     return energies, flats, collects

    def get_all_data_filenames(self):
        flats = [
            file.parent / file.name
            for file in self.scan_info["FILES"]
            if "ref_" in file.name
        ]
        projs = [
            file.parent / file.name
            for file in self.scan_info["FILES"]
            if "ref_" not in file.name
        ]
        return flats, projs

    def get_angles_from_filenames(self):
        reg_exp = re.compile("_[+-]\d\d\d.\d\d")
        self.angles_deg = map(
            reg_exp.findall, [str(file) for file in self.data_filenames]
        )
        self.angles_deg = [float(angle[0][1:]) for angle in self.angles_deg]
        seen = set()
        result = []
        for item in self.angles_deg:
            if item not in seen:
                seen.add(item)
                result.append(item)
        self.angles_deg = result
        self.angles_rad = [x * np.pi / 180 for x in self.angles_deg]

    def get_angles_from_metadata(self):
        self.angles_rad = [
            filemetadata["thetas"][0]
            for filemetadata in self.scan_info["PROJECTION_METADATA"]
        ]
        seen = set()
        result = []
        for item in self.angles_rad:
            if item not in seen:
                seen.add(item)
                result.append(item)
        self.angles_rad = result
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    def read_xrms_metadata(self, xrm_list):
        metadatas = []
        for i, filename in enumerate(xrm_list):
            ole = olefile.OleFileIO(str(filename))
            metadata = read_ole_metadata(ole)
            metadatas.append(metadata)
        return metadatas

    def load_xrms(self, xrm_list, Uploader):
        data_stack = None
        metadatas = []
        for i, filename in enumerate(xrm_list):
            data, metadata = read_xrm(str(filename))
            if data_stack is None:
                data_stack = np.zeros((len(xrm_list),) + data.shape, data.dtype)
            data_stack[i] = data
            metadatas.append(metadata)
            Uploader.upload_progress.value += 1
        data_stack = np.flip(data_stack, axis=1)
        return data_stack, metadatas

    def import_from_run_script(self, Uploader):
        all_collections = [[]]
        energies = [[self.selected_energies[0]]]
        with open(self.run_script_path, "r") as f:
            for line in f.readlines():
                if line.startswith("sete "):
                    energies.append(f"{float(line[5:]):08.2f}")
                    all_collections.append([])
                elif line.startswith("collect "):
                    filename = line[8:].strip()
                    all_collections[-1].append(self.run_script_path.parent / filename)
        if len(energies) > 1:
            energies.pop(0)
            all_collections.pop(0)
        else:
            energies = energies[0]

        for energy, collect in zip(energies, all_collections):
            if energy not in self.selected_energies:
                continue
            else:
                Uploader.upload_progress.value = 0
                self.current_energy_str = energy
                self.current_energy_float = float(energy)
                self.current_pixel_size = self.calculate_px_size(
                    float(energy), self.binning
                )
                Uploader.progress_output.clear_output()
                self.energy_label = Label(
                    f"{energy} eV", layout=Layout(justify_content="center")
                )
                with Uploader.progress_output:
                    display(Uploader.upload_progress)
                    display(self.energy_label)
                # Getting filename from specific energy
                self.flats_filenames = [
                    file.parent / file.name for file in collect if "ref_" in file.name
                ]
                self.data_filenames = [
                    file.parent / file.name
                    for file in collect
                    if "ref_" not in file.name
                ]
                self.proj_ind = [
                    True if "ref_" not in file.name else False for file in collect
                ]
                self.status_label = Label(
                    "Uploading .xrms.", layout=Layout(justify_content="center")
                )
                with Uploader.progress_output:
                    display(self.status_label)
                # Uploading Data
                Uploader.upload_progress.max = len(self.flats_filenames) + len(
                    self.data_filenames
                )

                self.flats, self.scan_info["FLAT_METADATA"] = self.load_xrms(
                    self.flats_filenames, Uploader
                )
                self._data, self.scan_info["PROJECTION_METADATA"] = self.load_xrms(
                    self.data_filenames, Uploader
                )
                self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
                self.energy_filedir = self.filedir / str(energy + "eV")
                if os.path.exists(self.energy_filedir):
                    shutil.rmtree(self.energy_filedir)
                os.makedirs(self.energy_filedir)
                self.status_label.value = "Saving flats."
                np.save(self.energy_filedir / "flats", self.flats)
                self.status_label.value = "Saving projections."
                np.save(self.energy_filedir / "projections", self._data)
                self.flats = None
                self._data = None
                self.flats = np.load(self.energy_filedir / "flats.npy", mmap_mode="r")
                self._data = np.load(
                    self.energy_filedir / "projections.npy", mmap_mode="r"
                )
                self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
                # Collecting information on where references are in the data stack
                self.status_label.value = "Calculating flat positions."
                copy_collect = collect.copy()
                i = 0
                for pos, file in enumerate(copy_collect):
                    if "ref_" in file.name:
                        if i == 0:
                            i = 1
                        elif i == 1:
                            copy_collect[pos] = 1
                    elif "ref_" not in file.name:
                        i = 0
                copy_collect = [value for value in copy_collect if value != 1]
                ref_ind = [
                    True if "ref_" in file.name else False for file in copy_collect
                ]
                ref_ind = [i for i in range(len(ref_ind)) if ref_ind[i]]
                ref_ind = sorted(list(set(ref_ind)))
                ref_ind = [ind - i for i, ind in enumerate(ref_ind)]
                # These indexes are at the position of self.data_filenames that
                # STARTS the next round after the references are taken
                self.flats_ind = ref_ind
                self.status_label.value = "Chunking data for dask normalization."
                projs = da.from_array(
                    self._data, chunks=(self.scan_info["NEXPOSURES"], -1, -1)
                ).astype(np.float32)
                flats = da.from_array(
                    self.flats, chunks=(self.scan_info["REFNEXPOSURES"], -1, -1)
                ).astype(np.float32)
                darks = da.from_array(self.darks, chunks=(-1, -1, -1)).astype(
                    np.float32
                )
                self.status_label.value = "Normalizing."
                self._data = RawProjectionsBase.normalize_and_average(
                    projs,
                    flats,
                    darks,
                    self.flats_ind,
                    self.scan_info["NEXPOSURES"],
                    status_label=self.status_label,
                )
                self.data = self._data
                self.status_label.value = "Saving projections as .npy for faster IO."
                self.save_normalized_as_npy(energy)
                self.saved_as_tiff = False
                if Uploader.save_tiff_on_import_checkbox.value:
                    self.status_label.value = "Saving projections as .tiff."
                    self.saved_as_tiff = True
                    self.save_normalized_as_tiff(energy)
                self.status_label.value = "Downsampling data for faster viewing."
                self._check_downsampled_data(energy)
                self.status_label.value = "Saving metadata."
                self.set_metadata()
                save_metadata(
                    self.energy_filedir / "import_metadata.json", self.metadata
                )

    # Groups each set of references and each set of projections together. Unused.
    def group_from_run_script(self):
        all_collections = [[]]
        energies = [[]]
        with open(self.run_script_path, "r") as f:
            for line in f.readlines():
                if line.startswith("sete "):
                    energies.append(f"{float(line[5:]):.2f}")
                    all_collections.append([])
                elif line.startswith("collect "):
                    filename = line[8:].strip()
                    all_collections[-1].append(self.run_script_path.parent / filename)
        all_collections.pop(0)
        energies.pop(0)
        for energy, collect in zip(energies, all_collections):
            if energy not in self.selected_energies:
                continue
            else:
                # getting all flats/projections
                ref_ind = [True if "ref_" in file.name else False for file in collect]
                i = 0
                copy_collect = collect.copy()
                for pos, file in enumerate(copy_collect):
                    if "ref_" in file.name:
                        if i == 0:
                            i = 1
                        elif i == 1:
                            copy_collect[pos] = 1
                    elif "ref_" not in file.name:
                        i = 0
                copy_collect = [value for value in copy_collect if value != 1]
                ref_ind = [
                    True if "ref_" in file.name else False for file in copy_collect
                ]
                ref_ind = [i for i in range(len(ref_ind)) if ref_ind[i]]
                self.ref_ind = ref_ind

                proj_ind = [
                    True if "ref_" not in file.name else False for file in collect
                ]
                self.flats_filenames = [
                    file.parent / file.name for file in collect if "ref_" in file.name
                ]
                self.data_filenames = [
                    file.parent / file.name
                    for file in collect
                    if "ref_" not in file.name
                ]
                # # intitializing switch statements
                files_grouped = [[]]
                file_type = ["reference"]
                i = 0
                adding_refs = True
                adding_projs = False
                for num, collection in enumerate(collect):
                    if ref_ind[num] and adding_refs:
                        files_grouped[-1].append(collection)
                    elif proj_ind[num] and ref_ind[num - 1]:
                        adding_refs = False
                        adding_projs = True
                        i = 0
                        files_grouped.append([])
                        files_grouped[-1].append(collection)
                        file_type.append("projection")
                    elif proj_ind[num - 1] and ref_ind[num]:
                        adding_refs = True
                        adding_projs = False
                        i = 0
                        files_grouped.append([])
                        files_grouped[-1].append(collection)
                        file_type.append("reference")
                    elif adding_projs and i < self.scan_info["NEXPOSURES"] - 1:
                        i += 1
                        files_grouped[-1].append(collection)
                    else:
                        i = 0
                        files_grouped.append([])
                        file_type.append("projection")

        return files_grouped, file_type
        # flats = [
        #     file.parent / file.name for file in collect if "ref_" in file.name
        # ]
        # projs = [
        #     file.parent / file.name
        #     for file in collect
        #     if "ref_" not in file.name
        # ]

    def metadata_to_DataFrame(self):

        # change metadata keys to be better looking
        keys = {
            "ENERGY": "Energy",
            "TOMO": "Tomo",
            "MOSAIC": "Mosaic",
            "MULTIEXPOSURE": "MultiExposure",
            "NREPEATSCAN": "Repeat Scan",
            "WAITNSECS": "Wait (s)",
            "NEXPOSURES": "Num. Exposures",
            "AVERAGEONTHEFLY": "Average On the Fly",
            "IMAGESPERPROJECTION": "Images/Projection",
            "REFNEXPOSURES": "Num. Ref Exposures",
            "REFEVERYEXPOSURES": "Ref/Num Exposures",
            "REFABBA": "Order",
            "REFDESPECKLEAVERAGE": "Ref Despeckle Avg",
            "APPLYREF": "Ref Applied",
            "MOSAICUP": "Up",
            "MOSAICDOWN": "Down",
            "MOSAICLEFT": "Left",
            "MOSAICRIGHT": "Right",
            "MOSAICOVERLAP": "Overlap (%)",
            "MOSAICCENTRALTILE": "Central Tile",
        }
        m = {keys[key]: self.scan_info[key] for key in keys}

        if m["Order"] == 0:
            m["Order"] = "ABAB"
        else:
            m["Order"] = "ABBA"

        # create headers and data for table
        middle_headers = []
        middle_headers.append(["Energy", "Tomo", "Mosaic", "MultiExposure"])
        middle_headers.append(
            [
                "Repeat Scan",
                "Wait (s)",
                "Num. Exposures",
                "Images/Projection",
            ]
        )
        middle_headers.append(
            ["Num. Ref Exposures", "Ref/Num Exposures", "Order", "Ref Despeckle Avg"]
        )
        middle_headers.append(["Up", "Down", "Left", "Right"])
        top_headers = []
        top_headers.append(["Layers"])
        top_headers.append(["Image Information"])
        top_headers.append(["Acquisition Information"])
        top_headers.append(["Reference Information"])
        top_headers.append(["Mosaic Information"])
        data = [
            [m[key] for key in middle_headers[i]] for i in range(len(middle_headers))
        ]
        middle_headers.insert(1, ["X Pixels", "Y Pixels", "Num. θ"])
        data.insert(1, [self.pxX, self.pxY, len(self.angles_rad)])

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


# https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
def safe_serialize(obj, f):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, f, default=default, indent=4)


def save_metadata(filename, metadata):
    with open(filename, "w+") as f:
        a = safe_serialize(metadata, f)


def load_metadata(filepath=None, filename=None, fullpath=None):
    if fullpath is not None:
        with open(fullpath) as f:
            metadata = json.load(f)
    else:
        fullpath = os.path.abspath(os.path.join(filepath, filename))
        with open(fullpath) as f:
            metadata = json.load(f)
    return metadata


def metadata_to_DataFrame(metadata):
    metadata_frame = {}
    time, title = parse_printed_time(metadata["analysis_time"])
    extra_headers = ["Prj X Range", "Prj Y Range", "Start Angle", "End Angle", title]
    metadata_frame["Headers"] = list(metadata["opts"].keys())
    metadata_frame["Headers"] = [
        metadata_frame["Headers"][i].replace("_", " ").title().replace("Num", "No.")
        for i in range(len(metadata_frame["Headers"]))
    ]
    metadata_frame["Headers"] = metadata_frame["Headers"] + extra_headers
    extra_values = [
        metadata["pixel_range_x"],
        metadata["pixel_range_y"],
        metadata["angle_start"],
        metadata["angle_end"],
        time,
    ]
    extra_values = [str(extra_values[i]) for i in range(len(extra_values))]
    metadata_frame["Values"] = [
        str(metadata["opts"][key]) for key in metadata["opts"]
    ] + extra_values
    metadata_frame = {
        metadata_frame["Headers"][i]: metadata_frame["Values"][i]
        for i in range(len(metadata_frame["Headers"]))
    }
    sr = pd.Series(metadata_frame)
    df = pd.DataFrame(sr).transpose()
    s = df.style.hide(axis="index")
    s.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
            {"selector": "td", "props": "text-align: center;" "font-size: 1.2em; "},
        ],
        overwrite=False,
    )
    return s


def parse_printed_time(timedict):
    if timedict["hours"] < 1:
        if timedict["minutes"] < 1:
            time = timedict["seconds"]
            title = "Time (s)"
        else:
            time = timedict["minutes"]
            title = "Time (min)"
    else:
        time = timedict["hours"]
        title = "Time (h)"

    time = f"{time:.1f}"
    return time, title

def rescale_parallel(i, images=None, ds_factor_list=None):
    print(i)
    print(ds_factor_list)
    return rescale(images[i], (ds_factor_list[1], ds_factor_list[2]))

def rescale_parallel_pool(n, images, ds_factor_list):
    with mp.Pool() as pool:
        rescale_partial = partial(rescale_parallel, images=images, ds_factor_list=ds_factor_list)
        return pool.map(rescale_partial, range(n))


# ARCHIVE:

# proj_ind = [
#     True if "ref_" not in file.name else False for file in collect
# ]
# flats_ind_positions = [i for i, val in enumerate(self.flats_ind) if val][
#     :: self.metadata["REFNEXPOSURES"]
# ]
# self.flats_ind = [
#     j for j in flats_ind_positions for i in range(self.metadata["REFNEXPOSURES"])
# ]