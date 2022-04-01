import numpy as np
import tifffile as tf
import tomopy.prep.normalize as tomopy_normalize
import os
import json
import dxchange
import re
import olefile
import pathlib
import tempfile
import dask.array as da
import dask
import shutil
import copy
import multiprocessing as mp
import pandas as pd
import time
import datetime
import h5py
import dask_image.imread

from abc import ABC, abstractmethod
from tomopy.sim.project import angles as angle_maker
from tomopyui.backend.util.dxchange.reader import read_ole_metadata, read_xrm
from tomopyui.backend.util.dask_downsample import pyramid_reduce_gaussian
from skimage.transform import rescale
from joblib import Parallel, delayed
from ipywidgets import *
from functools import partial


class IOBase:
    """
    Base class for all data imported. Contains some setter/getter attributes that also
    set other attributes, such setting number of pixels for a numpy array.

    Also has methods such as _check_downsampled_data, which checks for previously
    uploaded downsampled data, and writes it in a subfolder if not already there.

    _file_finder is under this class, but logically it does not belong here. TODO.
    """

    # Save keys
    normalized_projections_hdf_key = "normalized_projections.hdf5"
    normalized_projections_tif_key = "normalized_projections.tif"
    normalized_projections_npy_key = "normalized_projections.npy"

    # hdf keys
    hdf_key_raw_proj = "/exchange/data"
    hdf_key_raw_flats = "/exchange/data_white"
    hdf_key_raw_darks = "/exchange/data_dark"
    hdf_key_theta = "/exchange/theta"
    hdf_key_norm_proj = "/process/normalized/data"
    hdf_key_norm = "/process/normalized/"
    hdf_key_ds = "/process/downsampled/"
    hdf_key_ds_0 = "/process/downsampled/0/"
    hdf_key_ds_1 = "/process/downsampled/1/"
    hdf_key_ds_2 = "/process/downsampled/2/"
    hdf_key_data = "data"  # to be added after downsampled/0,1,2/...
    hdf_key_bin_frequency = "frequency"  # to be added after downsampled/0,1,2/...
    hdf_key_bin_centers = "bin_centers"  # to be added after downsampled/0,1,2/...
    hdf_key_image_range = "image_range"  # to be added after downsampled/0,1,2/...
    hdf_key_bin_edges = "bin_edges"
    hdf_key_percentile = "percentile"
    hdf_key_ds_factor = "ds_factor"
    hdf_key_process = "/process"

    hdf_keys_ds_hist = [
        hdf_key_bin_frequency,
        hdf_key_bin_centers,
        hdf_key_image_range,
        hdf_key_percentile,
    ]
    hdf_keys_ds_hist_scalar = [hdf_key_ds_factor]

    def __init__(self):

        self._data = np.random.rand(10, 100, 100)
        self.data = self._data
        self.data_ds = self.data
        self.imported = False
        self._filepath = pathlib.Path()
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
        self.hist = None
        self.allowed_extensions = [".npy", ".tiff", ".tif"]
        self.metadata = Metadata_General_Prenorm()
        self.hdf_file = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        (self.pxZ, self.pxY, self.pxX) = self._data.shape
        self.rangeX = (0, self.pxX - 1)
        self.rangeY = (0, self.pxY - 1)
        self.rangeZ = (0, self.pxZ - 1)
        self.size_gb = self._data.nbytes / 1048576 / 1000
        self.dtype = self._data.dtype
        self._data = value

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, value):
        self.filedir = value.parent
        self.filename = value.name
        self.extension = value.suffix
        self._filepath = value

    def _check_and_open_hdf(hdf_func):
        def inner_func(self, *args, **kwargs):
            self._filepath = self.filedir / self.filename
            if self.hdf_file:
                hdf_func(self, *args, **kwargs)
            else:
                self._open_hdf_file_read_only(self.filepath)

                hdf_func(self, *args, **kwargs)

        return inner_func

    def _check_and_open_hdf_read_write(hdf_func):
        def inner_func(self, *args, **kwargs):
            self._filepath = self.filedir / self.filename
            if self.hdf_file:
                hdf_func(self, *args, **kwargs)
            else:
                self._open_hdf_file_read_write(self.filepath)
                hdf_func(self, *args, **kwargs)

        return inner_func

    def _open_hdf_file_read_only(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self._close_hdf_file()
        self.hdf_file = h5py.File(filepath, "r")

    def _open_hdf_file_read_write(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self._close_hdf_file()
        self.hdf_file = h5py.File(filepath, "r+")

    def _open_hdf_file_append(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self._close_hdf_file()
        self.hdf_file = h5py.File(filepath, "a")

    @_check_and_open_hdf
    def _load_hdf_normalized_data_into_memory(self):
        self._data = self.hdf_file[self.hdf_key_norm_proj][:]
        self.data = self._data
        self.hist = {
            key: self.hdf_file[self.hdf_key_norm + key][:]
            for key in self.hdf_keys_ds_hist
        }
        pyramid_level = self.hdf_key_ds + str(0) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        self.data_ds = self.hdf_file[ds_data_key]

    @_check_and_open_hdf
    def _unload_hdf_normalized_and_ds(self):
        self._data = self.hdf_file[self.hdf_key_norm_proj]
        self.data = self._data
        self.hist = {
            key: self.hdf_file[self.hdf_key_norm + key] for key in self.hdf_keys_ds_hist
        }
        pyramid_level = self.hdf_key_ds + str(0) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        self.data_ds = self.hdf_file[ds_data_key]

    @_check_and_open_hdf
    def _load_hdf_ds_data_into_memory(self, pyramid_level=0):
        pyramid_level = self.hdf_key_ds + str(pyramid_level) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        self.data_ds = self.hdf_file[ds_data_key][:]
        self.hist = {
            key: self.hdf_file[pyramid_level + key][:] for key in self.hdf_keys_ds_hist
        }
        for key in self.hdf_keys_ds_hist_scalar:
            self.hist[key] = self.hdf_file[pyramid_level + key][()]
        self._data = self.hdf_file[self.hdf_key_norm_proj]
        self.data = self._data

    @_check_and_open_hdf
    def _delete_downsampled_data(self):
        if self.hdf_key_ds in self.hdf_file:
            del self.hdf_file[self.hdf_key_ds]

    def _close_hdf_file(self):
        if self.hdf_file:
            self.hdf_file.close()

    def _dask_hist(self):
        r = [da.min(self.data), da.max(self.data)]
        bins = 200 if self.data.size > 200 else self.data.size
        hist = da.histogram(self.data, range=r, bins=bins)
        percentile = da.percentile(self.data.flatten(), q=(0.5, 99.5))
        bin_edges = hist[1]
        return hist, r, bins, percentile

    def _dask_bin_centers(self, grp, write=False, savedir=None):
        tmp_filepath = copy.copy(self.filepath)
        self.filedir = savedir
        self.filepath = self.filedir / self.normalized_projections_hdf_key
        self._open_hdf_file_append()
        bin_edges = da.from_array(self.hdf_file[grp + self.hdf_key_bin_edges])
        bin_centers = da.from_array(
            [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        )
        if write and savedir is not None:
            data_dict = {grp + self.hdf_key_bin_centers: bin_centers}
            self.dask_data_to_h5(data_dict, savedir=savedir)
        self.filepath = tmp_filepath
        return bin_centers

    def _check_downsampled_data(self, label=None):
        """
        Checks to see if there is downsampled data in a directory. If it doesn't it
        will write new downsampled data.

        Parameters
        ----------
        energy : str, optional
            Energy in string format "\d\d\d\d\d.\d\d"
        label : widgets.Label, optional
            This is a label that will update in the frontend.
        """
        filedir = self.filedir
        files = [pathlib.Path(f).name for f in os.scandir(filedir) if not f.is_dir()]
        if self.normalized_projections_hdf_key in files:
            self._filepath = filedir / self.normalized_projections_hdf_key
            self.filepath = self._filepath
            self._open_hdf_file_read_write()
            if self.hdf_key_ds in self.hdf_file:
                self._load_hdf_ds_data_into_memory()
            else:
                pyramid_reduce_gaussian(
                    self.data,
                    io_obj=self,
                )
                self._load_hdf_ds_data_into_memory()

        else:
            try:
                self.filedir_ds = pathlib.Path(filedir / "downsampled").mkdir(
                    parents=True
                )
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
        """
        Writes downsampled data into folder using self.ds_vals.

        Parameters
        ----------
        self.ds_vals : list
            List of downsampling values to use for faster viewing. Currently, this is
            just [[1, 0.25, 0.25],]. List used to be longer, but deprecated in favor of
            only using one downsampling (for time savings).

        """
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in self.ds_vals]
        # TODO: make parallel on individual slices of data. see bottom of this .py (archived code)
        # ds_data = [rescale_parallel_pool(self.pxZ, self.data, ds_vals_list) for ds_vals_list in self.ds_vals]
        # :
        #     ds_data.append(rescale_parallel_pool(self.pxZ, self.data, self.ds_vals))
        # ds_data = Parallel(n_jobs=int(os.environ["num_cpu_cores"]))(
        #     delayed(rescale)(self.data, x) for x in self.ds_vals
        # )
        ds_data = rescale(self.data, self.ds_vals[0])
        # ds_data.append(self.data)
        for data, string in zip(ds_data, ds_vals_strs):
            np.save(self.filedir_ds / str("ds" + string), data)

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
        """
        Loads in downsampled data and their respective histograms to memory for
        fast viewing in the plotters.

        """
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in self.ds_vals]
        self.hist = [
            np.load(self.filedir_ds / str("ds" + string + "hist.npz"))
            for string in ds_vals_strs
        ]
        self.data_ds = [
            np.load(self.filedir_ds / str("ds" + string + ".npy"), mmap_mode="r")
            for string in ds_vals_strs
        ]

    def _file_finder(self, filedir, filetypes: list):
        """
        Used to find files of a given filetype in a directory. TODO: can go elsewhere.

        Parameters
        ----------
        filedir : pathlike, relative or absolute
            Folder in which to search for files.
        filetypes : list of str
            Filetypes list. e.g. [".txt", ".npy"]
        """
        files = [pathlib.PurePath(f) for f in os.scandir(filedir) if not f.is_dir()]
        files_with_ext = [
            file.name for file in files if any(x in file.name for x in filetypes)
        ]
        return files_with_ext

    def _file_finder_fullpath(self, filedir, filetypes: list):
        """
        Used to find files of a given filetype in a directory. TODO: can go elsewhere.

        Parameters
        ----------
        filedir : pathlike, relative or absolute
            Folder in which to search for files.
        filetypes : list of str
            Filetypes list. e.g. [".txt", ".npy"]
        """
        files = [pathlib.Path(f) for f in os.scandir(filedir) if not f.is_dir()]
        fullpaths_of_files_with_ext = [
            file for file in files if any(x in file.name for x in filetypes)
        ]
        return fullpaths_of_files_with_ext


class ProjectionsBase(IOBase, ABC):

    """
    Base class for projections data. Abstract methods include importing/exporting data
    and metadata. One can import a file directory, or a particular file within a
    directory. Aliases give options for users to extract their data with other keywords.

    """

    # https://stackoverflow.com/questions/4017572/
    # how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
    aliases = {
        "prj_imgs": "data",
        "num_angles": "pxZ",
        "width": "pxX",
        "height": "pxY",
        "px_range_x": "rangeX",
        "px_range_y": "rangeY",
        "px_range_z": "rangeZ",  # Could probably fix these
    }

    def __init__(self):
        super().__init__()
        self.ds_vals = [(1, 0.25, 0.25)]
        self.px_size = 1
        self.angles_rad = None
        self.angles_deg = None
        self.saved_as_tiff = False
        self.tiff_folder = False

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    def save_normalized_as_npy(self):
        """
        Saves current self.data under the current self.filedir as
        self.normalized_projections_npy_key
        """
        np.save(self.filedir / str(self.normalized_projections_npy_key), self.data)

    def save_normalized_as_tiff(self):
        """
        Saves current self.data under the current self.filedir as
        self.normalized_projections_tif_key
        """
        tf.imwrite(self.filedir / str(self.normalized_projections_tif_key), self.data)

    def dask_data_to_h5(self, data_dict, savedir=None):
        """
        Brings lazy dask arrays to hdf5 under /exchange under current filedir.

        Parameters
        ----------
        data_dict: dict
            Dictionary like {"/path/to/data": data}
        savedir: pathlib.Path
            Optional. Will default to self.filedir
        """
        if savedir is None:
            filedir = self.filedir
        else:
            filedir = savedir
        for key in data_dict:
            if not isinstance(data_dict[key], da.Array):
                data_dict[key] = da.from_array(data_dict[key])

        da.to_hdf5(
            filedir / self.normalized_projections_hdf_key,
            data_dict,
        )

    @abstractmethod
    def import_metadata(self, filedir):
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
        ...

    @abstractmethod
    def import_file_projections(self, filepath):
        ...


class Projections_Child(ProjectionsBase):
    def __init__(self, parent_projections):
        super().__init__()
        self.parent_projections = parent_projections
        # self.copy_from_parent()

    def copy_from_parent(self):
        self.parent_projections._unload_hdf_normalized_and_ds()
        self._data = self.parent_projections.data
        self.data = self._data
        self.data_ds = self.parent_projections.data_ds
        self.hist = self.parent_projections.hist
        self.hdf_file = self.parent_projections.hdf_file
        self.filedir = self.parent_projections.filedir
        self.filepath = self.parent_projections.filepath
        self.filename = self.parent_projections.filename

    def deepcopy_data_from_parent(self):
        self.parent_projections._load_hdf_normalized_data_into_memory()
        self._data = copy.deepcopy(self.parent_projections.data[:])
        self.data = self._data

    def import_file_projections(self):
        pass

    def import_filedir_projections(self):
        pass

    def import_metadata(self):
        pass


class Projections_Prenormalized(ProjectionsBase):
    """
    Prenormalized projections base class. This allows one to bring in data from a file
    directory or from a single tiff or npy, or multiple tiffs in a tiffstack. If the
    data has been normalized using tomopyui (at least for SSRL), importing from a
    folder will assume that "normalized_projections.npy" is in the folder, given that
    there is an import_metadata.json in that folder.

    Each method uses a parent Uploader instance for callbacks.

    """

    def import_filedir_projections(self, Uploader):
        """
        Similar process to import_file_projections. This one will be triggered if the
        tiff folder checkbox is selected on the frontend.
        """
        self.tic = time.perf_counter()
        Uploader.import_status_label.value = "Importing file directory."
        self.filedir = Uploader.filedir
        if not Uploader.imported_metadata:
            self.set_import_savedir(str(self.metadata.metadata["energy_str"] + "eV"))
            self.metadata.set_attributes_from_metadata_before_import(self)
        cwd = os.getcwd()
        os.chdir(self.filedir)
        self._data = dask_image.imread.imread("*.tif").astype(np.float32)
        self.data = self._data
        self.metadata.set_metadata_from_attributes_after_import(self)
        self.filedir = self.import_savedir
        self.save_data_and_metadata(Uploader)
        self._check_downsampled_data()
        os.chdir(cwd)

    def import_file_projections(self, Uploader):
        """
        Will import a file selected on the frontend. Goes through several steps:

        1. Set file directories/filenames/filepaths.
        2. Sets save directory
        3. Set projections attributes from the metadata supplied by frontend, if any.
        4. Determine what kind of file it is
        5. Upload
        6. Sets metadata from data info after upload
        7. Saves in the import directory.
        """
        self.tic = time.perf_counter()
        Uploader.import_status_label.value = "Importing single file."
        self.imported = False
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        self._filepath = self.filedir / self.filename
        self.filepath = self._filepath
        if not Uploader.imported_metadata:
            self.set_import_savedir(str(self.metadata.metadata["energy_str"] + "eV"))
            self.metadata.set_attributes_from_metadata_before_import(self)

        # if import metadata is found in the directory, self.normalized_projections_npy_key
        # will be uploaded. This behavior is probably OK if we stick to this file
        # structure
        if Uploader.imported_metadata:
            files = self._file_finder(self.filedir, [".npy", ".hdf5", ".h5"])
            if any([x in files for x in [".h5", ".hdf5"]]):
                Uploader.import_status_label.value = (
                    "Detected metadata and hdf5 file in this directory,"
                    + " uploading normalized_projections.hdf5"
                )
            elif ".npy" in files:
                Uploader.import_status_label.value = (
                    "Detected metadata and npy file in this directory,"
                    + " uploading normalized_projections.npy"
                )
                self.metadata.set_attributes_from_metadata(self)
                self._data = np.load(
                    self.filedir / "normalized_projections.npy"
                ).astype(np.float32)
            self.metadata.set_attributes_from_metadata(self)
            self._check_downsampled_data()
            if Uploader.save_tiff_on_import_checkbox.value:
                Uploader.import_status_label.value = "Saving projections as .tiff."
                self.saved_as_tiff = True
                self.save_normalized_as_tiff()
                self.metadata.metadata["saved_as_tiff"] = True
            self.imported = True

        elif any([x in self.filename for x in [".tif", ".tiff"]]):
            if self.tiff_folder:
                self.import_filedir_projections(Uploader)
                return
            self._data = np.array(
                dxchange.reader.read_tiff(self.filepath).astype(np.float32)
            )
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self.data = self._data
            self.metadata.set_metadata_from_attributes_after_import(self)
            self.save_data_and_metadata(Uploader)
            self.imported = True

        elif ".npy" in self.filename:
            self._data = np.load(self.filepath).astype(np.float32)
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self.data = self._data
            self.metadata.set_metadata_from_attributes_after_import(self)
            self.save_data_and_metadata(Uploader)
            self._check_downsampled_data()
            self.imported = True

    def get_img_shape(self, extension=None):
        """
        Gets the image shape of a tiff or npy with lazy loading.
        """

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
            with tf.TiffFile(self.filepath) as tif:
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
            size = np.load(self.filepath, mmap_mode="r").shape
            sizeZ = size[0]
            sizeY = size[1]
            sizeX = size[2]

        return (sizeZ, sizeY, sizeX)

    def make_angles(self):
        """
        Makes angles based on self.angle_start, self.angle_end, and self.pxZ.

        Also converts to degrees.
        """
        self.angles_rad = angle_maker(
            self.pxZ,
            ang1=self.angle_start,
            ang2=self.angle_end,
        )
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    def import_metadata(self, filepath):
        self.metadata = Metadata.parse_metadata_type(filepath)
        self.metadata.load_metadata()

    def set_import_savedir(self, folder_name):
        """
        Creates a save directory to put projections into.
        """
        self.import_savedir = pathlib.Path(self.filedir / folder_name)
        if self.import_savedir.exists():
            now = datetime.datetime.now()
            dt_str = now.strftime("%Y%m%d-%H%M-")
            save_name = dt_str + folder_name
            self.import_savedir = pathlib.Path(self.filedir / save_name)
            if self.import_savedir.exists():
                dt_str = now.strftime("%Y%m%d-%H%M%S-")
                save_name = dt_str + folder_name
                self.import_savedir = pathlib.Path(self.filedir / save_name)
        self.import_savedir.mkdir()
        self.filedir_ds = self.import_savedir / "downsampled"

    def save_data_and_metadata(self, Uploader):
        """
        Saves current data and metadata in import_savedir.
        """
        self.filedir = self.import_savedir
        self.dask_data_to_h5({self.hdf_key_norm_proj: self.data})
        self.saved_as_tiff = False
        if Uploader.save_tiff_on_import_checkbox.value:
            Uploader.import_status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata.metadata["saved_as_tiff"] = True
        self.metadata.filedir = self.filedir
        self.toc = time.perf_counter()
        self.metadata.metadata["import_time"] = self.toc - self.tic
        self.metadata.save_metadata()
        Uploader.import_status_label.value = "Checking for downsampled data."
        self._check_downsampled_data(label=Uploader.import_status_label)


class RawProjectionsBase(ProjectionsBase, ABC):
    """
    Base class for raw projections. Contains methods for normalization, and abstract
    methods that are required in any subclass of this class for data import. Some
    methods currently are not used by tomopyui (e.g. import_filedir_flats), but could
    be used in the future. If you don't want to use some of these methods, just
    write "def method(self):pass" to enable subclass instantiation.
    """

    def __init__(self):
        super().__init__()
        self.flats = None
        self.flats_ind = None
        self.darks = None
        self.normalized = False

    def check_import_savedir_exists(self, filedir_name):
        if self.import_savedir.exists():
            now = datetime.datetime.now()
            dt_str = now.strftime("%Y%m%d-%H%M-")
            save_name = dt_str + filedir_name
            self.import_savedir = pathlib.Path(self.filedir / save_name)
            if self.import_savedir.exists():
                dt_str = now.strftime("%Y%m%d-%H%M%S-")
                save_name = dt_str + filedir_name
                self.import_savedir = pathlib.Path(self.filedir / save_name)

    def normalize_nf(self):
        """
        Wrapper for tomopy's normalize_nf
        """
        self._data = tomopy_normalize.normalize_nf(
            self._data, self.flats, self.darks, self.flats_ind
        )
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    # for Tomopy-based normalization
    def normalize(self):
        """
        Wrapper for tomopy's normalize.
        """
        self._data = tomopy_normalize.normalize(self._data, self.flats, self.darks)
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    # For dask-based normalization (should be faster)
    def average_chunks(chunked_da):
        """
        Method required for averaging within the normalize_and_average method. Takes
        all chunks of a dask array

        Parameters
        ----------
        chunked_da: dask array
            Dask array with chunks to average over axis=0.
            For ex. if you have an initial numpy array with shape (50, 100, 100),
            and you chunk this along the 0th dimension into 5x (10, 100,100).
            The output of this function will be of dimension (5, 100, 100).

        Returns
        -------
        arr: dask array
            Dask array that has been averaged along axis=0 with respect to how many
            chunks it initially had.
        """

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
        projs,
        flats,
        dark,
        flat_loc,
        num_exposures_per_proj,
        status_label=None,
        compute=True,
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

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO

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

        # Don't know if putting @staticmethod above a decorator will mess it up, so this
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
        if compute:
            arr = arr.compute()

        return arr

    @staticmethod
    def normalize_no_locations_no_average(
        projs,
        flats,
        dark,
        status_label=None,
        compute=True,
    ):
        """
        Normalize using dask arrays. Only averages references and normalizes.
        """
        if status_label is not None:
            status_label.value = "Averaging flatfields."
        flat_mean = np.mean(flats, axis=0)
        dark = np.median(dark, axis=0)
        denominator = flat_mean - dark
        if status_label is not None:
            status_label.value = f"Dividing by flatfields and taking -log."
        projs = projs.rechunk({0: "auto", 1: -1, 2: -1})
        projs = projs / denominator
        projs = -da.log(projs)
        if compute:
            projs = projs.compute()
        return projs

    @abstractmethod
    def import_metadata(self, filedir):
        """
        After creating Metadata classes and subclasses, this logically belongs to those
        classes. TODO.
        """
        ...

    @abstractmethod
    def import_filedir_all(self, filedir):
        """
        Imports a directory of files containing raw projections, flats, and darks rather
        than a single file.
        """
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
        """
        Imports a directory of just the raw projections.
        """
        ...

    @abstractmethod
    def import_filedir_flats(self, filedir):
        """
        Imports a directory of just the raw flats.
        """
        ...

    @abstractmethod
    def import_filedir_darks(self, filedir):
        """
        Imports a directory of just the raw darks.
        """
        ...

    @abstractmethod
    def import_file_all(self, filepath):
        """
        Imports a file containing all raw projections, flats, and darks.
        """
        ...

    @abstractmethod
    def import_file_projections(self, filepath):
        """
        Imports a file containing all raw projections.
        """
        ...

    @abstractmethod
    def import_file_flats(self, filepath):
        """
        Imports a file containing all raw flats.
        """
        ...

    @abstractmethod
    def import_file_darks(self, filepath):
        """
        Imports a file containing all raw darks.
        """
        ...


class RawProjectionsXRM_SSRL62C(RawProjectionsBase):

    """
    Raw data import functions associated with SSRL 6-2c. If you specify a folder filled
    with raw XRMS, a ScanInfo file, and a run script, this will automatically import
    your data and save it in a subfolder corresponding to the energy.
    """

    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True
        self.metadata = Metadata_SSRL62C_Raw()

    def import_metadata(self, Uploader):
        self.metadata = Metadata_SSRL62C_Raw()
        self.data_hierarchy_level = 0
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
        self.scan_info["FLAT_METADATA"] = self.read_xrms_metadata(
            [self.flats_filenames[0]]
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
        self.metadata.filedir = Uploader.filedir
        self.metadata.filename = "raw_metadata.json"
        self.metadata.filepath = self.filedir / "raw_metadata.json"
        self.metadata.set_metadata(self)
        self.metadata.save_metadata()

    def import_filedir_all(self, Uploader):
        self.import_metadata(Uploader)
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

    def import_file_all(self, filepath):
        pass

    def import_file_projections(self, filepath):
        pass

    def import_file_flats(self, filepath):
        pass

    def import_file_darks(self, filepath):
        pass

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
        """
        Calculates the pixel size based on the energy and binning.
        From Johanna's calibration.
        """
        pixel_size = 0.002039449 * energy - 0.792164997
        pixel_size = pixel_size * binning
        return pixel_size

    def est_en_from_px_size(self, pixel_size, binning):
        """
        Estimates the energy based on the pixel size. This is for plain TOMO data where
        the energy is not available. You should be able to overwrite
        this in the frontend if energy cannot be found.
        Inverse of calculate_px_size.
        """
        # From Johanna's calibration doc
        energy = (pixel_size / binning + 0.792164997) / 0.002039449
        return energy

    def get_all_data_filenames(self):
        """
        Grabs the flats and projections filenames from scan info.

        Returns
        -------
        flats: list of pathlib.Path
            All flat file names in self.scan_info["FILES"]
        projs: list of pathlib.Path
            All projection file names in self.scan_info["FILES"]
        """

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
        """
        Grabs the angles from the file names in scan_info.
        """
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
        """
        Gets the angles from the raw image metadata.
        """
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
        """
        Reads XRM files and snags the metadata from them.

        Parameters
        ----------
        xrm_list: list(pathlib.Path)
            list of XRMs to grab metadata from
        Returns
        -------
        metadatas: list(dict)
            List of metadata dicts for files in xrm_list
        """
        metadatas = []
        for i, filename in enumerate(xrm_list):
            ole = olefile.OleFileIO(str(filename))
            metadata = read_ole_metadata(ole)
            metadatas.append(metadata)
        return metadatas

    def load_xrms(self, xrm_list, Uploader):
        """
        Loads XRM data from a file list in order, concatenates them to produce a stack
        of data (npy).

        Parameters
        ----------
        xrm_list: list(pathlib.Path)
            list of XRMs to upload
        Uploader: `Uploader`
            Should have an upload_progress attribute. This is the progress bar.
        Returns
        -------
        data_stack: np.ndarray()
            Data grabbed from xrms in xrm_list
        metadatas: list(dict)
            List of metadata dicts for files in xrm_list
        """
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
        """
        Script to upload selected data from a run script.

        If an energy is selected on the frontend, it will be added to the queue to
        upload and normalize.

        This reads the run script in the folder. Each time "set e" is in the run script,
        this means that the energy is changing and signifies a new tomography.

        Parameters
        ----------
        Uploader: `Uploader`
            Should have an upload_progress, status_label, and progress_output attribute.
            This is for the progress bar and information during the upload progression.
        """
        all_collections = [[]]
        energies = [[self.selected_energies[0]]]
        parent_metadata = self.metadata.metadata.copy()
        if "data_hierarchy_level" not in parent_metadata:
            try:
                with open(self.filepath) as f:
                    parent_metadata = json.load(
                        self.run_script_path.parent / "raw_metadata.json"
                    )
            except Exception:
                pass
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
                _tmp_filedir = copy.deepcopy(self.filedir)
                self.metadata = Metadata_SSRL62C_Prenorm()
                self.metadata.set_parent_metadata(parent_metadata)
                Uploader.upload_progress.value = 0
                self.energy_str = energy
                self.energy_float = float(energy)
                self.px_size = self.calculate_px_size(float(energy), self.binning)
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
                energy_filedir_name = str(energy + "eV")
                self.import_savedir = self.filedir / energy_filedir_name
                # TODO clean this with method
                if self.import_savedir.exists():
                    now = datetime.datetime.now()
                    dt_str = now.strftime("%Y%m%d-%H%M-")
                    save_name = dt_str + energy_filedir_name
                    self.import_savedir = pathlib.Path(self.filedir / save_name)
                    if self.import_savedir.exists():
                        dt_str = now.strftime("%Y%m%d-%H%M%S-")
                        save_name = dt_str + energy_filedir_name
                        self.import_savedir = pathlib.Path(self.filedir / save_name)
                self.import_savedir.mkdir()
                projs, flats, darks = self.setup_normalize()
                self.status_label.value = "Calculating flat positions."
                self.flats_ind_from_collect(collect)
                self.status_label.value = "Normalizing."
                self._data = RawProjectionsBase.normalize_and_average(
                    projs,
                    flats,
                    darks,
                    self.flats_ind,
                    self.scan_info["NEXPOSURES"],
                    status_label=self.status_label,
                    compute=False,
                )
                self.data = self._data

                self.status_label.value = "Saving projections as .npy for faster IO."
                hist, r, bins, percentile = self._dask_hist()
                grp = IOBase.hdf_key_norm + "/"
                data_dict = {
                    self.hdf_key_norm_proj: self.data,
                    grp + self.hdf_key_bin_frequency: hist[0],
                    grp + self.hdf_key_bin_edges: hist[1],
                    grp + self.hdf_key_image_range: r,
                    grp + self.hdf_key_percentile: percentile,
                }
                self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
                self._dask_bin_centers(grp, write=True, savedir=self.import_savedir)
                self.saved_as_tiff = False
                self.filedir = self.import_savedir
                if Uploader.save_tiff_on_import_checkbox.value:
                    self.status_label.value = "Saving projections as .tiff."
                    self.saved_as_tiff = True
                    self.save_normalized_as_tiff()
                self.status_label.value = "Downsampling data for faster viewing."
                self._check_downsampled_data()
                self.status_label.value = "Saving metadata."
                self.data_hierarchy_level = 1
                self.metadata.set_metadata(self)
                self.metadata.filedir = self.import_savedir
                self.metadata.filename = "import_metadata.json"
                self.metadata.save_metadata()
                self.filedir = _tmp_filedir
                self._close_hdf_file()

    def setup_normalize(self):
        """
        Function to lazy load flats and projections as npy, convert to chunked dask
        arrays for normalization.

        Returns
        -------
        projs: dask array
            Projections chunked by scan_info["NEXPOSURES"]
        flats: dask array
            References chunked by scan_info["REFNEXPOSURES"]
        darks: dask array
            Zeros array with the same image dimensions as flats
        """
        data_dict = {
            self.hdf_key_raw_flats: self.flats,
            self.hdf_key_raw_proj: self._data,
        }
        self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
        self.filepath = self.import_savedir / self.normalized_projections_hdf_key
        self._open_hdf_file_read_write()
        z_chunks_proj = self.scan_info["NEXPOSURES"]
        z_chunks_flats = self.scan_info["REFNEXPOSURES"]
        self.flats = None
        self._data = None

        self.flats = da.from_array(
            self.hdf_file[self.hdf_key_raw_flats],
            chunks=(z_chunks_flats, -1, -1),
        ).astype(np.float32)

        self._data = da.from_array(
            self.hdf_file[self.hdf_key_raw_proj],
            chunks=(z_chunks_proj, -1, -1),
        ).astype(np.float32)
        darks = da.from_array(self.darks, chunks=(-1, -1, -1)).astype(np.float32)
        projs = self._data
        flats = self.flats

        return projs, flats, darks

    def flats_ind_from_collect(self, collect):
        """
        Calculates where the flats indexes are based on the current "collect", which
        is a collection under each "set e" from the run script importer.

        This will set self.flats_ind for normalization.
        """
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
        ref_ind = [True if "ref_" in file.name else False for file in copy_collect]
        ref_ind = [i for i in range(len(ref_ind)) if ref_ind[i]]
        ref_ind = sorted(list(set(ref_ind)))
        ref_ind = [ind - i for i, ind in enumerate(ref_ind)]
        # These indexes are at the position of self.data_filenames that
        # STARTS the next round after the references are taken
        self.flats_ind = ref_ind

    def string_num_totuple(self, s):
        """
        Helper function for import_metadata. I forget what it does. :)
        """
        return (
            "".join(c for c in s if c.isalpha()) or None,
            "".join(c for c in s if c.isdigit() or None),
        )


class RawProjectionsTiff_SSRL62B(RawProjectionsBase):

    """
    Raw data import functions associated with SSRL 6-2c. If you specify a folder filled
    with raw XRMS, a ScanInfo file, and a run script, this will automatically import
    your data and save it in a subfolder corresponding to the energy.
    """

    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True
        self.metadata_projections = Metadata_SSRL62B_Raw_Projections()
        self.metadata_references = Metadata_SSRL62B_Raw_References()
        self.metadata = Metadata_SSRL62B_Raw(
            self.metadata_projections, self.metadata_references
        )

    def import_data(self, Uploader):
        self.metadata_projections.set_extra_metadata(Uploader)
        self.metadata_references.set_extra_metadata(Uploader)
        # self.metadata_projections.filedir = Uploader.
        self.metadata.filedir = self.metadata_projections.filedir
        self.filedir = self.metadata.filedir
        self.metadata.filepath = self.metadata.filedir / self.metadata.filename
        self.metadata.save_metadata()
        save_filedir_name = str(self.metadata_projections.metadata["energy_str"] + "eV")
        self.import_savedir = self.metadata_projections.filedir / save_filedir_name
        self.check_import_savedir_exists(save_filedir_name)
        self.import_savedir.mkdir()
        self.import_filedir_projections(Uploader)
        self.import_filedir_flats(Uploader)
        projs, flats, darks = self.setup_normalize(Uploader)
        Uploader.import_status_label.value = "Normalizing projections"
        self._data = self.normalize_no_locations_no_average(
            projs, flats, darks, compute=False
        )
        self.data = self._data
        self.filedir = self.import_savedir
        da.to_hdf5(
            self.filedir / self.normalized_projections_hdf_key,
            "/projections/data",
            self.data,
        )
        Uploader.import_status_label.value = "Getting data from hdf5"
        open_file = h5py.File(self.filedir / self.normalized_projections_hdf_key)[
            "projections"
        ]["data"]
        self._data = da.from_array(open_file)
        self.data = self._data
        Uploader.import_status_label.value = "Downsampling data in a pyramid"
        self._check_downsampled_data(label=Uploader.import_status_label)
        self.metadata_projections.set_attributes_from_metadata(self)
        self.metadata_prenorm = Metadata_SSRL62B_Prenorm()
        self.metadata_prenorm.set_metadata(self)
        self.metadata_prenorm.metadata[
            "parent_metadata"
        ] = self.metadata.metadata.copy()
        self.metadata_prenorm.filedir = self.filedir
        self.metadata_prenorm.filepath = self.filedir / self.metadata_prenorm.filename
        self.metadata_prenorm.save_metadata()
        self.metadata_prenorm.save_metadata_h5(
            open_file,
        )

    def import_metadata(self):
        self.metadata = Metadata_SSRL62B_Raw(
            self.metadata_projections, self.metadata_references
        )

    def import_metadata_projections(self, Uploader):
        self.projections_filedir = Uploader.projections_metadata_filepath.parent
        self.metadata_projections = Metadata_SSRL62B_Raw_Projections()
        self.metadata_projections.filedir = (
            Uploader.projections_metadata_filepath.parent
        )
        self.metadata_projections.filename = Uploader.projections_metadata_filepath.name
        self.metadata_projections.filepath = Uploader.projections_metadata_filepath
        self.metadata_projections.parse_raw_metadata()
        self.metadata_projections.set_extra_metadata(Uploader)

    def import_metadata_references(self, Uploader):
        self.references_filedir = Uploader.references_metadata_filepath.parent
        self.metadata_references = Metadata_SSRL62B_Raw_References()
        self.metadata_references.filedir = Uploader.references_metadata_filepath.parent
        self.metadata_references.filename = Uploader.references_metadata_filepath.name
        self.metadata_references.filepath = Uploader.references_metadata_filepath
        self.metadata_references.parse_raw_metadata()
        self.metadata_references.set_extra_metadata(Uploader)

    def import_filedir_all(self, Uploader):
        pass

    def import_filedir_projections(self, Uploader):
        tifffiles = self.metadata_projections.metadata["filenames"]
        tifffiles = [self.projections_filedir / file for file in tifffiles]
        Uploader.upload_progress.value = 0
        Uploader.upload_progress.max = len(tifffiles)
        Uploader.import_status_label.value = "Uploading projections"
        Uploader.progress_output.clear_output()
        with Uploader.progress_output:
            display(
                VBox(
                    [Uploader.upload_progress, Uploader.import_status_label],
                    layout=Layout(justify_content="center", align_items="center"),
                )
            )

        arr = []
        for file in tifffiles:
            arr.append(tf.imread(file))
            Uploader.upload_progress.value += 1
        Uploader.import_status_label.value = "Converting to numpy array"
        arr = np.array(arr)
        arr = np.rot90(arr, axes=(1, 2))
        Uploader.import_status_label.value = "Converting to dask array"
        arr = da.from_array(arr, chunks={0: "auto", 1: -1, 2: -1})
        Uploader.import_status_label.value = "Saving in normalized_projections.hdf5"
        da.to_hdf5(
            self.import_savedir / self.normalized_projections_hdf_key,
            "/raw/projections",
            arr,
        )

    def import_filedir_flats(self, Uploader):
        tifffiles = self.metadata_references.metadata["filenames"]
        tifffiles = [self.metadata_references.filedir / file for file in tifffiles]
        Uploader.upload_progress.value = 0
        Uploader.upload_progress.max = len(tifffiles)
        Uploader.import_status_label.value = "Uploading references"
        arr = []
        for file in tifffiles:
            arr.append(tf.imread(file))
            Uploader.upload_progress.value += 1
        Uploader.import_status_label.value = "Converting to numpy array"
        arr = np.array(arr)
        arr = np.rot90(arr, axes=(1, 2))
        Uploader.import_status_label.value = "Converting to dask array"
        arr = da.from_array(arr, chunks={0: "auto", 1: -1, 2: -1})
        Uploader.import_status_label.value = "Saving in normalized_projections.hdf5"
        da.to_hdf5(
            self.import_savedir / self.normalized_projections_hdf_key,
            "/raw/flats",
            arr,
        )

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, filepath):
        pass

    def import_file_projections(self, filepath):
        pass

    def import_file_flats(self, filepath):
        pass

    def import_file_darks(self, filepath):
        pass

    def setup_normalize(self, Uploader):
        """
        Function to lazy load flats and projections as npy, convert to chunked dask
        arrays for normalization.

        Returns
        -------
        projs: dask array
            Projections chunked by scan_info["NEXPOSURES"]
        flats: dask array
            References chunked by scan_info["REFNEXPOSURES"]
        darks: dask array
            Zeros array with the same image dimensions as flats
        """
        self.flats = None
        self._data = None
        open_file = h5py.File(
            self.import_savedir / self.normalized_projections_hdf_key, "a"
        )
        self.flats = open_file["raw"]["flats"]
        self._data = open_file["raw"]["projections"]
        self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
        projs = da.from_array(self._data).astype(np.float32)
        flats = da.from_array(self.flats).astype(np.float32)
        darks = da.from_array(self.darks).astype(np.float32)
        return projs, flats, darks


class RawProjectionsHDF5_ALS832(RawProjectionsBase):
    """
    This class holds your projections data, metadata, and functions associated with
    importing that data and metadata.

    For SSRL62C, this is a very complicated class. Because of your h5 data storage,
    it is relatively more straightforward to import and normalize.

    You can overload the functions in subclasses if you have more complicated
    import and normalization protocols for your data.
    """

    def __init__(self):
        super().__init__()
        self.allowed_extensions = [".h5"]
        self.metadata = Metadata_ALS_832_Raw()

    def import_filedir_all(self, filedir):
        pass

    def import_filedir_projections(self, filedir):
        pass

    def import_filedir_flats(self, filedir):
        pass

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, Uploader):
        self.import_status_label = Uploader.import_status_label
        self.tic = time.perf_counter()
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        self.filepath = self.filedir / self.filename
        self.metadata = Uploader.reset_metadata_to()
        self.metadata.load_metadata_h5(self.filepath)
        self.metadata.set_attributes_from_metadata(self)
        self.import_status_label.value = "Importing"
        self.metadata.set_attributes_from_metadata(self)
        (
            self._data,
            self.flats,
            self.darks,
            self.angles_rad,
        ) = dxchange.exchange.read_aps_tomoscan_hdf5(self.filepath)
        self.data = self._data
        self.angles_deg = (180 / np.pi) * self.angles_rad
        self.metadata.set_metadata(self)
        self.metadata.save_metadata()
        self.imported = True
        self.import_savedir = self.filedir / str(self.filepath.stem)
        # if the save directory already exists (you have previously uploaded this
        # raw data), then it will create a datestamped folder.
        if self.import_savedir.exists():
            now = datetime.datetime.now()
            dt_str = now.strftime("%Y%m%d-%H%M-")
            save_name = dt_str + str(self.filepath.stem)
            self.import_savedir = pathlib.Path(self.filedir / save_name)
            if self.import_savedir.exists():
                dt_str = now.strftime("%Y%m%d-%H%M%S-")
                save_name = dt_str + str(self.filepath.stem)
                self.import_savedir = pathlib.Path(self.filedir / save_name)
        self.import_savedir.mkdir()
        self.import_status_label.value = "Normalizing"
        self.normalize()
        _metadata = self.metadata.metadata.copy()
        self.import_status_label.value = "Saving projections as npy for faster IO"
        self.filedir = self.import_savedir
        self.save_normalized_as_npy()
        self._check_downsampled_data()
        self.toc = time.perf_counter()
        self.metadata = self.save_normalized_metadata(self.toc - self.tic, _metadata)

    def import_metadata(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.metadata.load_metadata_h5(filepath)
        self.metadata.set_attributes_from_metadata(self)

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

    def save_normalized_metadata(self, import_time=None, parent_metadata=None):
        metadata = Metadata_ALS_832_Prenorm()
        metadata.filedir = self.filedir
        metadata.metadata = parent_metadata.copy()
        if parent_metadata is not None:
            metadata.metadata["parent_metadata"] = parent_metadata.copy()
        if import_time is not None:
            metadata.metadata["import_time"] = import_time
        metadata.set_metadata(self)
        metadata.save_metadata()
        return metadata


class RawProjectionsHDF5_APS(RawProjectionsHDF5_ALS832):
    """
    See RawProjectionsHDF5_ALS832 superclass description.
    # Francesco: you may need to edit here.
    """

    def __init__(self):
        super().__init__()
        self.metadata = Metadata_APS_Raw()

    def save_normalized_metadata(self, import_time=None, parent_metadata=None):
        metadata = Metadata_APS_Prenorm()
        metadata.filedir = self.filedir
        metadata.metadata = parent_metadata.copy()
        if parent_metadata is not None:
            metadata.metadata["parent_metadata"] = parent_metadata.copy()
        if import_time is not None:
            metadata.metadata["import_time"] = import_time
        metadata.set_metadata(self)
        metadata.save_metadata()
        return metadata


class Metadata(ABC):
    """
    Base class for all metadatas.
    """

    def __init__(self):
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }
        self.table_label = Label(style=self.header_font_style)
        self.metadata = {}
        self.filedir = None
        self.filename = None
        self.filepath = None

    def save_metadata(self):
        with open(self.filedir / self.filename, "w+") as f:
            a = safe_serialize(self.metadata, f)

    def load_metadata(self):
        with open(self.filepath) as f:
            self.metadata = json.load(f)

        return self.metadata

    def set_parent_metadata(self, parent_metadata):
        self.metadata["parent_metadata"] = parent_metadata
        self.metadata["data_hierarchy_level"] = (
            parent_metadata["data_hierarchy_level"] + 1
        )

    def create_metadata_box(self):
        """
        Creates the box to be displayed on the frontend when importing data. Has both
        a label and the metadata dataframe (stored in table_output).

        """
        self.metadata_to_DataFrame()
        self.table_output = Output()
        if self.dataframe is not None:
            with self.table_output:
                display(self.dataframe)
        self.metadata_vbox = VBox(
            [self.table_label, self.table_output], layout=Layout(align_items="center")
        )

    @staticmethod
    def parse_metadata_type(filepath: pathlib.Path = None, metadata=None):
        """
        Determines the type of metadata by looking at the "metadata_type" key in the
        loaded dictionary.

        Parameters
        ----------
        filepath: pathlib.Path
            Filepath for the metadata. If this is not specified, metadata should be
            specified
        metadata: dict
            A metadata dictionary with the "metadata_type" key. If this is not
            specified, a filepath should be specified.

        Returns
        -------
        A metadata instance with the metadata.

        """
        if filepath is not None:
            with open(filepath) as f:
                metadata = json.load(f)

        if "metadata_type" not in metadata:
            metadata["metadata_type"] = "SSRL62C_Normalized"

        # General Data
        if metadata["metadata_type"] == "General_Normalized":
            metadata_instance = Metadata_General_Prenorm()

        # SSRL Beamlines
        if metadata["metadata_type"] == "SSRL62C_Normalized":
            metadata_instance = Metadata_SSRL62C_Prenorm()
        if metadata["metadata_type"] == "SSRL62C_Raw":
            metadata_instance = Metadata_SSRL62C_Raw()
        if metadata["metadata_type"] == "SSRL62B_Normalized":
            metadata_instance = Metadata_SSRL62B_Prenorm()
        if metadata["metadata_type"] == "SSRL62B_Raw":
            metadata_instance = Metadata_SSRL62B_Raw()

        # ALS Beamlines
        if metadata["metadata_type"] == "ALS832_Normalized":
            metadata_instance = Metadata_ALS_832_Prenorm()
        if metadata["metadata_type"] == "ALS832_Raw":
            metadata_instance = Metadata_ALS_832_Raw()

        # Metadata through rest of processing pipeline
        if metadata["metadata_type"] == "Prep":
            metadata_instance = Metadata_Prep()
        if metadata["metadata_type"] == "Align":
            metadata_instance = Metadata_Align()
        if metadata["metadata_type"] == "Recon":
            metadata_instance = Metadata_Recon()

        if filepath is not None:
            metadata_instance.filedir = filepath.parent
            metadata_instance.filename = filepath.name
            metadata_instance.filepath = filepath

        return metadata_instance

    @staticmethod
    def get_metadata_hierarchy(filepath):
        """
        Reads in a metadata file from filepath and determines its hierarchy. Generates
        a list of `Metadata` instances, found by Metadata.parse_metadata_type.

        Parameters
        ----------
        filepath: pathlike
            Metadata file path.

        Returns
        -------
        metadata_insts: list(`Metadata`)
            List of metadata instances associated with the metadata file.
        """
        with open(filepath) as f:
            metadata = json.load(f)
        num_levels = metadata["data_hierarchy_level"]
        metadata_insts = []
        for i in range(num_levels + 1):
            metadata_insts.append(
                Metadata.parse_metadata_type(metadata=metadata.copy())
            )
            metadata_insts[i].metadata = metadata.copy()
            if "parent_metadata" in metadata:
                metadata = metadata["parent_metadata"].copy()
        return metadata_insts

    @abstractmethod
    def set_metadata(self, projections):
        """
        Sets metadata from projections attributes.
        """
        ...

    @abstractmethod
    def metadata_to_DataFrame(self):
        """
        This will take the metadata that you have and turn it into a table for display
        on the frontend. It is a little complicated, but I don't know pandas very well.
        You will have "top_headers" which are the headers at the top of the table like
        "Image Information". The subheaders are called "middle_headers": things like
        the X Pixels, Y Pixels, and the number of angles. Then below each of the middle
        headers, you have the data. The dimensions of each should match up properly

        This creates a dataframe and then s.set_table_styles() styles it. This styling
        function is based on CSS, which I know very little about. You can make the
        table as fancy as you want, but for now I just have a blue background header
        and white lines dividing the major table sections.
        """
        ...


class Metadata_General_Prenorm(Metadata):
    """
    General prenormalized metadata. This will be created if you are importing a tiff
    or tiff stack, or npy file that was not previously imported using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "General_Normalized"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.imported = False
        self.table_label.value = "User Metadata"

    def set_metadata(self, projections):
        pass

    def metadata_to_DataFrame(self):
        # create headers and data for table
        self.metadata["energy_str"] = f"{self.metadata['energy_float']:0.2f}"
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        ang_res = self.metadata["angular_resolution"]
        self.metadata["num_angles"] = int((end_angle - start_angle) / ang_res)

        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                "Angular Resolution (°)": f"{ang_res:0.2f}",
            },
            {
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Binning": self.metadata["binning"],
                "Num. θ (est)": self.metadata["num_angles"],
            },
        ]
        if "pxX" in self.metadata:
            self.metadata_list_for_table[1]["X Pixels"] = self.metadata["pxX"]
            self.metadata_list_for_table[1]["Y Pixels"] = self.metadata["pxY"]
            self.metadata_list_for_table[1]["Num. θ"] = self.metadata["pxZ"]
            self.make_angles_from_metadata()

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
                # ("Acquisition Information", middle_headers[0][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s

    def set_attributes_from_metadata_before_import(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy = projections.energy_float
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]

    def set_metadata_from_attributes_after_import(self, projections):
        self.metadata["normalized_projections_size_gb"] = projections.size_gb
        self.metadata["normalized_projections_directory"] = str(
            projections.import_savedir
        )
        if "filedir_ds" in projections.__dict__:
            self.metadata["downsampled_projections_directory"] = str(
                projections.filedir_ds
            )
        self.metadata["downsampled_values"] = projections.ds_vals
        self.metadata["saved_as_tiff"] = projections.saved_as_tiff
        self.metadata["num_angles"] = projections.data.shape[0]
        self.metadata["pxX"] = projections.data.shape[2]
        self.metadata["pxY"] = projections.data.shape[1]
        self.metadata["pxZ"] = projections.data.shape[0]

    def set_attributes_from_metadata(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy = projections.energy_float
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        projections.size_gb = self.metadata["normalized_projections_size_gb"]
        projections.import_savedir = pathlib.Path(
            self.metadata["normalized_projections_directory"]
        )
        if "downsampled_projections_directory" in self.metadata:
            projections.filedir_ds = pathlib.Path(
                self.metadata["downsampled_projections_directory"]
            )
        projections.ds_vals = self.metadata["downsampled_values"]
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]
        if "angles_rad" in self.metadata:
            projections.angles_rad = self.metadata["angles_rad"]
            projections.angles_deg = self.metadata["angles_deg"]

    def make_angles_from_metadata(self):
        self.metadata["angles_rad"] = angle_maker(
            self.metadata["pxZ"],
            ang1=self.metadata["start_angle"],
            ang2=self.metadata["end_angle"],
        )
        self.metadata["angles_rad"] = list(self.metadata["angles_rad"])
        self.metadata["angles_deg"] = [
            x * 180 / np.pi for x in self.metadata["angles_rad"]
        ]


class Metadata_SSRL62C_Raw(Metadata):
    """
    Raw metadata from SSRL 6-2C. Will be created if you import a folder filled with
    raw XRMs.
    """

    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62C_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2C Raw Metadata"

    def set_attributes_from_metadata(self, projections):
        pass

    def set_metadata(self, projections):
        self.metadata["scan_info"] = copy.deepcopy(projections.scan_info)
        self.metadata["scan_info"]["FILES"] = [
            str(file) for file in projections.scan_info["FILES"]
        ]
        self.metadata["scan_info_path"] = str(projections.scan_info_path)
        self.metadata["run_script_path"] = str(projections.run_script_path)
        self.metadata["flats_filenames"] = [
            str(file) for file in projections.flats_filenames
        ]
        self.metadata["projections_filenames"] = [
            str(file) for file in projections.data_filenames
        ]
        self.metadata["scan_type"] = projections.scan_type
        self.metadata["scan_order"] = projections.scan_order
        self.metadata["pxX"] = projections.pxX
        self.metadata["pxY"] = projections.pxY
        self.metadata["pxZ"] = projections.pxZ
        self.metadata["num_angles"] = projections.pxZ
        self.metadata["angles_rad"] = projections.angles_rad
        self.metadata["angles_deg"] = projections.angles_deg
        self.metadata["start_angle"] = float(projections.angles_deg[0])
        self.metadata["end_angle"] = float(projections.angles_deg[-1])
        self.metadata["binning"] = projections.binning
        self.metadata["projections_exposure_time"] = projections.scan_info[
            "PROJECTION_METADATA"
        ][0]["exposure_time"]
        self.metadata["references_exposure_time"] = projections.scan_info[
            "FLAT_METADATA"
        ][0]["exposure_time"]
        self.metadata["all_raw_energies_float"] = projections.energies_list_float
        self.metadata["all_raw_energies_str"] = projections.energies_list_str
        self.metadata["all_raw_pixel_sizes"] = projections.raw_pixel_sizes
        self.metadata[
            "pixel_size_from_scan_info"
        ] = projections.pixel_size_from_metadata
        self.metadata["energy_units"] = "eV"
        self.metadata["pixel_units"] = "nm"
        self.metadata["raw_projections_dtype"] = str(projections.raw_data_type)
        self.metadata["raw_projections_directory"] = str(
            projections.data_filenames[0].parent
        )
        self.metadata["data_hierarchy_level"] = projections.data_hierarchy_level

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
        m = {keys[key]: self.metadata["scan_info"][key] for key in keys}

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
        data.insert(
            1,
            [
                self.metadata["pxX"],
                self.metadata["pxY"],
                len(self.metadata["angles_rad"]),
            ],
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
                ],
                ("Image Information", "X Pixels"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Mosaic Information", "Up"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s


class Metadata_SSRL62C_Prenorm(Metadata_SSRL62C_Raw):
    """
    Metadata class for data from SSRL 6-2C that was normalized using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "SSRL62C_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.table_label.value = "SSRL 6-2C TomoPyUI-Imported Metadata"

    def set_metadata(self, projections):
        super().set_metadata(projections)
        metadata_to_remove = [
            "scan_info_path",
            "run_script_path",
            "scan_info",
            "scan_type",
            "scan_order",
            "all_raw_energies_float",
            "all_raw_energies_str",
            "all_raw_pixel_sizes",
            "pixel_size_from_scan_info",
            "raw_projections_dtype",
        ]
        # removing unneeded things from parent raw
        [
            self.metadata.pop(name)
            for name in metadata_to_remove
            if name in self.metadata
        ]
        self.metadata["flats_ind"] = projections.flats_ind
        self.metadata["user_overwrite_energy"] = projections.user_overwrite_energy
        self.metadata["energy_str"] = projections.energy_str
        self.metadata["energy_float"] = projections.energy_float
        self.metadata["pixel_size"] = projections.px_size
        self.metadata["normalized_projections_dtype"] = str(np.dtype(np.float32))
        self.metadata["normalized_projections_size_gb"] = projections.size_gb
        self.metadata["normalized_projections_directory"] = str(
            projections.import_savedir
        )
        self.metadata[
            "normalized_projections_filename"
        ] = projections.normalized_projections_hdf_key
        self.metadata["normalization_function"] = "dask"
        self.metadata["saved_as_tiff"] = projections.saved_as_tiff

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        exp_time_proj = f"{self.metadata['projections_exposure_time']:0.2f}"
        exp_time_ref = f"{self.metadata['references_exposure_time']:0.2f}"
        # ds_vals = self.metadata["downsampled_values"]
        # ds_vals = [x[2] for x in ds_vals]
        if self.metadata["user_overwrite_energy"]:
            user_overwrite = "Yes"
        else:
            user_overwrite = "No"
        if self.metadata["saved_as_tiff"]:
            save_as_tiff = "Yes"
        else:
            save_as_tiff = "No"
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                # "Scan Type": self.metadata["scan_type"],
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
                "Energy Overwritten": user_overwrite,
                ".tif Saved": save_as_tiff,
                # "Downsample Values": ds_vals,
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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Other Information", middle_headers[2][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s

    def set_attributes_from_metadata(self, projections):
        # projections.scan_info = copy.deepcopy(self.metadata["scan_info"])
        # projections.scan_info["FILES"] = [
        #     pathlib.Path(file) for file in self.metadata["scan_info"]["FILES"]
        # ]
        # projections.scan_info_path = self.metadata["scan_info_path"]
        # projections.run_script_path = pathlib.Path(self.metadata["run_script_path"])
        projections.flats_filenames = [
            pathlib.Path(file) for file in self.metadata["flats_filenames"]
        ]
        projections.data_filenames = [
            pathlib.Path(file) for file in self.metadata["projections_filenames"]
        ]
        # projections.scan_type = self.metadata["scan_type"]
        # projections.scan_order = self.metadata["scan_order"]
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        # projections.energies_list_float = self.metadata["all_raw_energies_float"]
        # projections.energies_list_str = self.metadata["all_raw_energies_str"]
        projections.user_overwrite_energy = self.metadata["user_overwrite_energy"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_units = self.metadata["energy_units"]
        # projections.raw_pixel_sizes = self.metadata["all_raw_pixel_sizes"]
        # projections.pixel_size_from_metadata = self.metadata[
        #     "pixel_size_from_scan_info"
        # ]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        # projections.size_gb = self.metadata["normalized_projections_size_gb"]
        projections.import_savedir = pathlib.Path(
            self.metadata["normalized_projections_directory"]
        )
        if "downsampled_projections_directory" in self.metadata:
            projections.filedir_ds = pathlib.Path(
                self.metadata["downsampled_projections_directory"]
            )
        if "flats_ind" in self.metadata:
            projections.flats_ind = self.metadata["flats_ind"]
        if "downsampled_values" in self.metadata:
            projections.ds_vals = self.metadata["downsampled_values"]
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]


class Metadata_SSRL62B_Raw_Projections(Metadata):
    """
    Raw projections metadata from SSRL 6-2B.
    """

    summary_key = "Summary"
    coords_default_key = r"Coords-Default/"
    metadata_default_key = r"Metadata-Default/"

    def __init__(self):
        super().__init__()
        self.loaded_metadata = False  # did we load metadata yet? no
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw_Projections"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw Projections Metadata"

    def parse_raw_metadata(self):
        self.load_metadata()
        self.summary = self.imported_metadata["Summary"].copy()
        self.metadata["acquisition_name"] = self.summary["Prefix"]
        self.metadata["angular_resolution"] = self.summary["z-step_um"] / 1000
        self.metadata["pxZ"] = self.summary["Slices"]
        self.metadata["num_angles"] = self.metadata["pxZ"]
        self.metadata["pixel_type"] = self.summary["PixelType"]
        self.meta_keys = [
            key for key in self.imported_metadata.keys() if "Metadata-Default" in key
        ]
        self.metadata["angles_deg"] = [
            self.imported_metadata[key]["ZPositionUm"] / 1000 for key in self.meta_keys
        ]
        self.metadata["angles_rad"] = [
            x * np.pi / 180 for x in self.metadata["angles_deg"]
        ]
        self.metadata["start_angle"] = self.metadata["angles_deg"][0]
        self.metadata["end_angle"] = self.metadata["angles_deg"][-1]
        self.metadata["exposure_times_ms"] = [
            self.imported_metadata[key]["Exposure-ms"] for key in self.meta_keys
        ]
        self.metadata["average_exposure_time"] = np.mean(
            self.metadata["exposure_times_ms"]
        )
        self.metadata["elapsed_times_ms"] = [
            self.imported_metadata[key]["ElapsedTime-ms"] for key in self.meta_keys
        ]
        self.metadata["received_times"] = [
            self.imported_metadata[key]["ReceivedTime"] for key in self.meta_keys
        ]
        self.metadata["filenames"] = [
            key.replace(r"Metadata-Default/", "") for key in self.meta_keys
        ]
        self.metadata["widths"] = [
            self.imported_metadata[key]["Width"] for key in self.meta_keys
        ]
        self.metadata["heights"] = [
            self.imported_metadata[key]["Height"] for key in self.meta_keys
        ]
        self.metadata["binnings"] = [
            self.imported_metadata[key]["Binning"] for key in self.meta_keys
        ]
        self.metadata["pxX"] = self.metadata["heights"][0]
        self.metadata["pxY"] = self.metadata["widths"][0]
        self.loaded_metadata = True

    def set_extra_metadata(self, Uploader):
        self.metadata["energy_float"] = Uploader.energy_textbox.value
        self.metadata["energy_str"] = f"{self.metadata['energy_float']:0.2f}"
        self.metadata["energy_units"] = Uploader.energy_units_dropdown.value
        self.metadata["pixel_size"] = Uploader.px_size_textbox.value
        self.metadata["pixel_units"] = Uploader.px_units_dropdown.value

    def load_metadata(self):
        with open(self.filepath) as f:
            self.imported_metadata = json.load(f)
        return self.imported_metadata

    def set_attributes_from_metadata(self, projections):
        projections.num_angles = self.metadata["num_angles"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.start_angle = self.metadata["start_angle"]
        projections.pxZ = self.metadata["pxZ"]
        projections.pxY = self.metadata["pxY"]
        projections.pxX = self.metadata["pxX"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_units = self.metadata["energy_units"]
        projections.pixel_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]

    def set_metadata(self, projections):
        pass

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        # ds_vals = [x[2] for x in ds_vals]
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                "Exp. Time (ms)": f"{self.metadata['average_exposure_time']:0.2f}",
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. θ": self.metadata["num_angles"],
                "Binning": self.metadata["binnings"][0],
            }
            # {
            # ".tif Saved": save_as_tiff,
            # "Downsample Values": ds_vals,
            # },
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
        # top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                # ("Other Information", middle_headers[2][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
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

        self.dataframe = s


class Metadata_SSRL62B_Raw_References(Metadata_SSRL62B_Raw_Projections):
    """
    Raw reference metadata from SSRL 6-2B.
    """

    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw_References"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw References Metadata"

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        # ds_vals = [x[2] for x in ds_vals]
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                # "Start θ (°)": f"{start_angle:0.1f}",
                # "End θ (°)": f"{end_angle:0.1f}",
                "Exp. Time (ms)": f"{self.metadata['average_exposure_time']:0.2f}",
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. Refs": len(self.metadata["widths"]),
                "Binning": self.metadata["binnings"][0],
            },
            # {
            # ".tif Saved": save_as_tiff,
            # "Downsample Values": ds_vals,
            # },
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
        # top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                # ("Other Information", middle_headers[2][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
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

        self.dataframe = s


class Metadata_SSRL62B_Raw(Metadata_SSRL62B_Raw_Projections):
    """
    Raw reference metadata from SSRL 6-2B.
    """

    def __init__(self, metadata_projections, metadata_references):
        super().__init__()
        self.metadata_projections = metadata_projections
        self.metadata_references = metadata_references
        self.metadata["projections_metadata"] = self.metadata_projections.metadata
        self.metadata["references_metadata"] = self.metadata_references.metadata
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw Metadata"

    def metadata_to_DataFrame(self):
        # create headers and data for table
        self.metadata_projections.create_metadata_box()
        self.metadata_references.create_metadata_box()

    def create_metadata_hbox(self):
        """
        Creates the box to be displayed on the frontend when importing data. Has both
        a label and the metadata dataframe (stored in table_output).

        """
        self.metadata_to_DataFrame()
        self.table_output = Output()
        if (
            self.metadata_projections.dataframe is not None
            and self.metadata_references.dataframe is not None
        ):
            self.metadata_hbox = HBox(
                [
                    self.metadata_projections.metadata_vbox,
                    self.metadata_references.metadata_vbox,
                ],
                layout=Layout(justify_content="center"),
            )


class Metadata_SSRL62B_Prenorm(Metadata_SSRL62B_Raw_Projections):
    """
    Metadata class for data from SSRL 6-2C that was normalized using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "SSRL62C_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.table_label.value = "SSRL 6-2B TomoPyUI-Imported Metadata"

    def set_metadata(self, projections):
        self.metadata["num_angles"] = projections.num_angles
        self.metadata["angles_deg"] = projections.angles_deg
        self.metadata["angles_rad"] = projections.angles_rad
        self.metadata["start_angle"] = projections.start_angle
        self.metadata["end_angle"] = projections.end_angle
        self.metadata["start_angle"] = projections.start_angle
        self.metadata["pxZ"] = projections.pxZ
        self.metadata["pxY"] = projections.pxY
        self.metadata["pxX"] = projections.pxX
        self.metadata["energy_float"] = projections.energy_float
        self.metadata["energy_str"] = projections.energy_str
        self.metadata["energy_units"] = projections.energy_units
        self.metadata["pixel_size"] = projections.pixel_size
        self.metadata["pixel_units"] = projections.pixel_units

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
        if self.metadata["user_overwrite_energy"]:
            user_overwrite = "Yes"
        else:
            user_overwrite = "No"
        if self.metadata["saved_as_tiff"]:
            save_as_tiff = "Yes"
        else:
            save_as_tiff = "No"
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                # "Scan Type": self.metadata["scan_type"],
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
                "Energy Overwritten": user_overwrite,
                ".tif Saved": save_as_tiff,
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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Other Information", middle_headers[2][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s

    def set_attributes_from_metadata(self, projections):
        # projections.scan_info = copy.deepcopy(self.metadata["scan_info"])
        # projections.scan_info["FILES"] = [
        #     pathlib.Path(file) for file in self.metadata["scan_info"]["FILES"]
        # ]
        # projections.scan_info_path = self.metadata["scan_info_path"]
        # projections.run_script_path = pathlib.Path(self.metadata["run_script_path"])
        projections.flats_filenames = [
            pathlib.Path(file) for file in self.metadata["flats_filenames"]
        ]
        projections.data_filenames = [
            pathlib.Path(file) for file in self.metadata["projections_filenames"]
        ]
        # projections.scan_type = self.metadata["scan_type"]
        # projections.scan_order = self.metadata["scan_order"]
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        # projections.energies_list_float = self.metadata["all_raw_energies_float"]
        # projections.energies_list_str = self.metadata["all_raw_energies_str"]
        projections.user_overwrite_energy = self.metadata["user_overwrite_energy"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_units = self.metadata["energy_units"]
        # projections.raw_pixel_sizes = self.metadata["all_raw_pixel_sizes"]
        # projections.pixel_size_from_metadata = self.metadata[
        #     "pixel_size_from_scan_info"
        # ]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        # projections.size_gb = self.metadata["normalized_projections_size_gb"]
        projections.import_savedir = pathlib.Path(
            self.metadata["normalized_projections_directory"]
        )
        projections.filedir_ds = pathlib.Path(
            self.metadata["downsampled_projections_directory"]
        )
        if "flats_ind" in self.metadata:
            projections.flats_ind = self.metadata["flats_ind"]
        projections.ds_vals = self.metadata["downsampled_values"]
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]


class Metadata_ALS_832_Raw(Metadata):
    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.table_label.value = "ALS 8.3.2 Metadata"

    def set_metadata(self, projections):

        self.metadata["numslices"] = projections.pxY
        self.metadata["numrays"] = projections.pxX
        self.metadata["num_angles"] = projections.pxZ
        self.metadata["pxsize"] = projections.px_size
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = projections.propagation_dist
        self.metadata["propagation_dist_units"] = "mm"
        self.metadata["angularrange"] = projections.angular_range
        self.metadata["kev"] = projections.energy
        self.metadata["energy_units"] = "keV"
        if projections.angles_deg is not None:
            self.metadata["angles_deg"] = list(projections.angles_deg)
            self.metadata["angles_rad"] = list(projections.angles_rad)

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.propagation_dist = self.metadata["propagation_dist"]
        projections.propagation_dist_units = "mm"
        projections.angular_range = self.metadata["angularrange"]
        projections.energy = self.metadata["kev"]
        projections.units = self.metadata["energy_units"]

    def load_metadata_h5(self, h5_filepath):
        self.filedir = h5_filepath.parent
        self.filepath = h5_filepath
        self.metadata["pxY"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.metadata["numslices"] = self.metadata["pxY"]
        self.metadata["pxX"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.metadata["numrays"] = self.metadata["pxX"]
        self.metadata["pxZ"] = int(
            dxchange.read_hdf5(h5_filepath, "/process/acquisition/rotation/num_angles")[
                0
            ]
        )
        self.metadata["num_angles"] = self.metadata["pxZ"]
        self.metadata["pxsize"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/pixel_size"
            )[0]
            / 10.0
        )  # /10 to convert units from mm to cm
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = dxchange.read_hdf5(
            h5_filepath,
            "/measurement/instrument/camera_motor_stack/setup/camera_distance",
        )[1]
        self.metadata["energy_float"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/monochromator/energy"
            )[0]
            / 1000
        )
        self.metadata["kev"] = self.metadata["energy_float"]
        self.metadata["energy_str"] = str(self.metadata["energy_float"])
        self.metadata["energy_units"] = "keV"
        self.metadata["angularrange"] = dxchange.read_hdf5(
            h5_filepath, "/process/acquisition/rotation/range"
        )[0]

    def metadata_to_DataFrame(self):

        # create headers and data for table
        top_headers = []
        middle_headers = []
        data = []
        # Image information
        top_headers.append(["Image Information"])
        middle_headers.append(["X Pixels", "Y Pixels", "Num. θ"])
        data.append(
            [
                self.metadata["numrays"],
                self.metadata["numslices"],
                self.metadata["num_angles"],
            ]
        )

        top_headers.append(["Experiment Settings"])
        middle_headers.append(
            ["Energy (keV)", "Propagation Distance (mm)", "Angular range (deg)"]
        )
        data.append(
            [
                self.metadata["kev"],
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
                ("Experiment Settings", "Energy (keV)"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s


class Metadata_ALS_832_Prenorm(Metadata_ALS_832_Raw):
    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.data_hierarchy_level = self.metadata["data_hierarchy_level"]
        self.table_label.value = ""

    def set_metadata(self, projections):
        super().set_metadata(projections)
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Normalized"
        self.metadata["data_hierarchy_level"] = 1

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.energy = self.metadata["kev"] / 1000
        projections.units = "eV"
        projections.angles_deg = self.metadata["angles_deg"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angle_start = projections.angles_rad[0]
        projections.angle_end = projections.angles_rad[-1]

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):
        """
        Method overloaded because the metadata table is the same as the superclass.
        This avoids a space between tables during display.
        """
        self.metadata_vbox = Output()


class Metadata_APS_Raw(Metadata):
    # Francesco: you will need to edit here.
    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "APS_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.table_label.value = "APS Metadata"

    def set_metadata(self, projections):
        """
        Sets metadata from the APS h5 filetype
        """
        self.metadata["numslices"] = projections.pxY
        self.metadata["numrays"] = projections.pxX
        self.metadata["num_angles"] = projections.pxZ
        self.metadata["pxsize"] = projections.px_size
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = projections.propagation_dist
        self.metadata["propagation_dist_units"] = "mm"
        self.metadata["angularrange"] = projections.angular_range
        self.metadata["kev"] = projections.energy
        self.metadata["energy_units"] = "keV"
        if projections.angles_deg is not None:
            self.metadata["angles_deg"] = list(projections.angles_deg)
            self.metadata["angles_rad"] = list(projections.angles_rad)

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.propagation_dist = self.metadata["propagation_dist"]
        projections.propagation_dist_units = "mm"
        projections.angular_range = self.metadata["angularrange"]
        projections.energy = self.metadata["kev"]
        projections.units = self.metadata["energy_units"]

    def load_metadata_h5(self, h5_filepath):
        """
        Loads in metadata from h5 file. You can probably use your dxchange function
        to read all the metadata in at once. Not sure how it works for you.

        The keys in the self.metadata dictionary can be whatever you want, as long as
        your set_attributes_from_metadata function above sets the values correctly.
        """
        # set metadata filepath to the filepath above
        self.filedir = h5_filepath.parent
        self.filepath = h5_filepath

        # Here you will set your metadata. I have left these here from the ALS metadata
        # class for reference. Some things are not inside the metadata (i.e.
        # "energy_units") that I set manually.
        self.metadata["pxY"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.metadata["numslices"] = self.metadata["pxY"]
        self.metadata["pxX"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.metadata["numrays"] = self.metadata["pxX"]
        self.metadata["pxZ"] = int(
            dxchange.read_hdf5(h5_filepath, "/process/acquisition/rotation/num_angles")[
                0
            ]
        )
        self.metadata["num_angles"] = self.metadata["pxZ"]
        self.metadata["pxsize"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/pixel_size"
            )[0]
            / 10.0
        )  # /10 to convert units from mm to cm
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = dxchange.read_hdf5(
            h5_filepath,
            "/measurement/instrument/camera_motor_stack/setup/camera_distance",
        )[1]
        self.metadata["energy_float"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/monochromator/energy"
            )[0]
            / 1000
        )
        self.metadata["kev"] = self.metadata["energy_float"]
        self.metadata["energy_str"] = str(self.metadata["energy_float"])
        self.metadata["energy_units"] = "keV"
        self.metadata["angularrange"] = dxchange.read_hdf5(
            h5_filepath, "/process/acquisition/rotation/range"
        )[0]

    def metadata_to_DataFrame(self):

        # create headers and data for table
        top_headers = []
        middle_headers = []
        data = []
        # Image information
        top_headers.append(["Image Information"])
        middle_headers.append(["X Pixels", "Y Pixels", "Num. θ"])
        data.append(
            [
                self.metadata["numrays"],
                self.metadata["numslices"],
                self.metadata["num_angles"],
            ]
        )

        top_headers.append(["Experiment Settings"])
        middle_headers.append(
            ["Energy (keV)", "Propagation Distance (mm)", "Angular range (deg)"]
        )
        data.append(
            [
                self.metadata["kev"],
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
                ("Experiment Settings", "Energy (keV)"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
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

        self.dataframe = s


class Metadata_APS_Prenorm(Metadata_APS_Raw):
    """
    Prenormalized metadata class. The table produced by this function may look nearly
    the same for you. For the SSRL version, it looks very different because there is a
    lot of excess information that I store in the SSRL raw metadata file.

    It is important to have this because "import_metadata.json" will be stored in a
    subfolder of the parent, raw data.

    Because the APS prenormalized metadata table looks identical to the raw metadata
    table, I overloaded the create_metadata_box() function to be just an Output widget.

    You can get as fancy as you want with this.

    # Francesco: you will need to edit here.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "APS_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.data_hierarchy_level = self.metadata["data_hierarchy_level"]
        self.table_label.value = ""

    def set_metadata(self, projections):
        super().set_metadata(projections)
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Normalized"
        self.metadata["data_hierarchy_level"] = 1

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.energy = self.metadata["kev"] / 1000
        projections.units = "eV"
        projections.angles_deg = self.metadata["angles_deg"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angle_start = projections.angles_rad[0]
        projections.angle_end = projections.angles_rad[-1]

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):
        """
        Method overloaded because the metadata table is the same as the superclass.
        This avoids a space between tables during display.
        """
        self.metadata_vbox = Output()


class Metadata_Prep(Metadata):
    def __init__(self):
        super().__init__()
        self.table_label.value = "Preprocessing Methods"
        self.prep_list_label_style = {
            "font_size": "16px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }

    def set_metadata(self, Prep):
        self.metadata["metadata_type"] = "Prep"
        self.filename = "prep_metadata.json"
        self.parent_metadata = Prep.Import.projections.metadata
        self.metadata["parent_metadata"] = self.parent_metadata.metadata
        if "data_hierarchy_level" in self.parent_metadata.metadata:
            self.metadata["data_hierarchy_level"] = (
                self.parent_metadata.metadata["data_hierarchy_level"] + 1
            )
        else:
            self.metadata["data_hierarchy_level"] = 2
        self.metadata["prep_list"] = [
            (x[1].method_name, x[1].opts) for x in Prep.prep_list
        ]
        self.table_label.value = "Preprocessing Metadata"

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):
        display_str = [x[0] + " → " for x in self.metadata["prep_list"][:-1]]
        display_str = "".join(display_str + [self.metadata["prep_list"][-1][0]])

        self.prep_list_label = Label(display_str, style=self.prep_list_label_style)
        self.metadata_vbox = VBox(
            [self.table_label, self.prep_list_label],
            layout=Layout(align_items="center"),
        )

    def set_attributes_from_metadata(self, projections):
        pass


class Metadata_Align(Metadata):
    """
    Works with both Align and TomoAlign instances.
    """

    def __init__(self):
        super().__init__()
        self.metadata["opts"] = {}
        self.metadata["methods"] = {}
        self.metadata["save_opts"] = {}
        self.table_label.value = "Alignment Metadata"

    def set_metadata(self, Align):
        self.metadata["metadata_type"] = "Align"
        self.metadata["opts"]["downsample"] = Align.downsample
        self.metadata["opts"]["downsample_factor"] = Align.ds_factor
        self.metadata["opts"]["num_iter"] = Align.num_iter
        self.metadata["opts"]["center"] = Align.center
        self.metadata["opts"]["pad"] = (
            Align.paddingX,
            Align.paddingY,
        )
        self.metadata["opts"]["extra_options"] = Align.extra_options
        self.metadata["methods"] = Align.methods_opts
        self.metadata["save_opts"] = Align.save_opts
        self.metadata["px_range_x"] = Align.px_range_x
        self.metadata["px_range_y"] = Align.px_range_y
        self.metadata["parent_filedir"] = Align.projections.filedir
        self.metadata["parent_filename"] = Align.projections.filename
        self.metadata["angle_start"] = Align.projections.angles_deg[0]
        self.metadata["angle_end"] = Align.projections.angles_deg[-1]
        self.set_metadata_obj_specific(Align)

    def set_metadata_obj_specific(self, Align):
        self.metadata["opts"]["upsample_factor"] = Align.upsample_factor
        self.metadata["opts"]["pre_alignment_iters"] = Align.pre_alignment_iters
        self.metadata["use_subset_correlation"] = Align.use_subset_correlation
        self.metadata["subset_range_x"] = Align.subset_range_x
        self.metadata["subset_range_y"] = Align.subset_range_y
        self.metadata["opts"]["num_batches"] = Align.num_batches

    def metadata_to_DataFrame(self):
        metadata_frame = {}
        time, title = parse_printed_time(self.metadata["analysis_time"])
        extra_headers = [
            "Prj X Range",
            "Prj Y Range",
            "Start Angle",
            "End Angle",
            title,
        ]
        metadata_frame["Headers"] = list(self.metadata["opts"].keys())
        metadata_frame["Headers"] = [
            metadata_frame["Headers"][i].replace("_", " ").title().replace("Num", "No.")
            for i in range(len(metadata_frame["Headers"]))
        ]
        metadata_frame["Headers"] = metadata_frame["Headers"] + extra_headers
        extra_values = [
            self.metadata["px_range_x"],
            self.metadata["px_range_y"],
            self.metadata["angle_start"],
            self.metadata["angle_end"],
            time,
        ]
        extra_values = [str(extra_values[i]) for i in range(len(extra_values))]
        metadata_frame["Values"] = [
            str(self.metadata["opts"][key]) for key in self.metadata["opts"]
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
                {
                    "selector": "th:not(.index_name)",
                    "props": "background-color: #0F52BA; color: white;",
                },
            ],
            overwrite=False,
        )

        self.dataframe = s

    def set_attributes_from_metadata(self, Align):
        Align.downsample = self.metadata["opts"]["downsample"]
        Align.ds_factor = self.metadata["opts"]["downsample_factor"]
        Align.num_iter = self.metadata["opts"]["num_iter"]
        Align.center = self.metadata["opts"]["center"]
        (Align.paddingX, Align.paddingY) = self.metadata["opts"]["pad"]
        Align.pad = (Align.paddingX, Align.paddingY)
        Align.extra_options = self.metadata["opts"]["extra_options"]
        Align.methods_opts = self.metadata["methods"]
        Align.save_opts = self.metadata["save_opts"]
        if "px_range_x" in self.metadata.keys():
            Align.px_range_x = self.metadata["px_range_x"]
            Align.px_range_y = self.metadata["px_range_y"]
        else:
            Align.px_range_x = self.metadata["pixel_range_x"]
            Align.px_range_y = self.metadata["pixel_range_y"]
        self.set_attributes_object_specific(Align)

    def set_attributes_object_specific(self, Align):
        Align.upsample_factor = self.metadata["opts"]["upsample_factor"]
        Align.pre_alignment_iters = self.metadata["opts"]["pre_alignment_iters"]
        Align.subset_x = self.metadata["subset_range_x"]
        Align.subset_y = self.metadata["subset_range_y"]
        Align.use_subset_correlation = self.metadata["use_subset_correlation"]
        Align.num_batches = self.metadata["opts"]["num_batches"]


class Metadata_Recon(Metadata_Align):
    def set_metadata(self, Recon):
        super().set_metadata(Recon)
        self.metadata["metadata_type"] = "Recon"
        self.table_label.value = "Reconstruction Metadata"

    def set_metadata_obj_specific(self, Recon):
        pass

    def set_attributes_from_metadata(self, Recon):
        super().set_attributes_from_metadata(Recon)

    def set_attributes_object_specific(self, Recon):
        pass


# https://stackoverflow.com/questions/
# 51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
def safe_serialize(obj, f):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, f, default=default, indent=4)


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
    return rescale(images[i], (ds_factor_list[1], ds_factor_list[2]))


def rescale_parallel_pool(n, images, ds_factor_list):
    with mp.Pool() as pool:
        rescale_partial = partial(
            rescale_parallel, images=images, ds_factor_list=ds_factor_list
        )
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


# # Groups each set of references and each set of projections together. Unused.
# def group_from_run_script(self):
#     all_collections = [[]]
#     energies = [[]]
#     with open(self.run_script_path, "r") as f:
#         for line in f.readlines():
#             if line.startswith("sete "):
#                 energies.append(f"{float(line[5:]):.2f}")
#                 all_collections.append([])
#             elif line.startswith("collect "):
#                 filename = line[8:].strip()
#                 all_collections[-1].append(self.run_script_path.parent / filename)
#     all_collections.pop(0)
#     energies.pop(0)
#     for energy, collect in zip(energies, all_collections):
#         if energy not in self.selected_energies:
#             continue
#         else:
#             # getting all flats/projections
#             ref_ind = [True if "ref_" in file.name else False for file in collect]
#             i = 0
#             copy_collect = collect.copy()
#             for pos, file in enumerate(copy_collect):
#                 if "ref_" in file.name:
#                     if i == 0:
#                         i = 1
#                     elif i == 1:
#                         copy_collect[pos] = 1
#                 elif "ref_" not in file.name:
#                     i = 0
#             copy_collect = [value for value in copy_collect if value != 1]
#             ref_ind = [
#                 True if "ref_" in file.name else False for file in copy_collect
#             ]
#             ref_ind = [i for i in range(len(ref_ind)) if ref_ind[i]]
#             self.ref_ind = ref_ind

#             proj_ind = [
#                 True if "ref_" not in file.name else False for file in collect
#             ]
#             self.flats_filenames = [
#                 file.parent / file.name for file in collect if "ref_" in file.name
#             ]
#             self.data_filenames = [
#                 file.parent / file.name
#                 for file in collect
#                 if "ref_" not in file.name
#             ]
#             # # intitializing switch statements
#             files_grouped = [[]]
#             file_type = ["reference"]
#             i = 0
#             adding_refs = True
#             adding_projs = False
#             for num, collection in enumerate(collect):
#                 if ref_ind[num] and adding_refs:
#                     files_grouped[-1].append(collection)
#                 elif proj_ind[num] and ref_ind[num - 1]:
#                     adding_refs = False
#                     adding_projs = True
#                     i = 0
#                     files_grouped.append([])
#                     files_grouped[-1].append(collection)
#                     file_type.append("projection")
#                 elif proj_ind[num - 1] and ref_ind[num]:
#                     adding_refs = True
#                     adding_projs = False
#                     i = 0
#                     files_grouped.append([])
#                     files_grouped[-1].append(collection)
#                     file_type.append("reference")
#                 elif adding_projs and i < self.scan_info["NEXPOSURES"] - 1:
#                     i += 1
#                     files_grouped[-1].append(collection)
#                 else:
#                     i = 0
#                     files_grouped.append([])
#                     file_type.append("projection")

#     return files_grouped, file_type
