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
import scipy.ndimage as ndi

from abc import ABC, abstractmethod
from tomopy.sim.project import angles as angle_maker
from tomopyui.backend.util.dxchange.reader import read_ole_metadata, read_xrm, read_txrm
from tomopyui.backend.util.dask_downsample import pyramid_reduce_gaussian
from skimage.transform import rescale
from joblib import Parallel, delayed
from ipywidgets import *
from functools import partial
from tomopyui.backend.io.metadata import Metadata_General_Prenorm, Metadata

# if os.environ["cuda_enabled"] == "True":
#     from tomopyui.widgets.prep import shift_projections



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
        # load normalized data into memory
        self._data = self.hdf_file[self.hdf_key_norm_proj][:]
        self.data = self._data
        pyramid_level = self.hdf_key_ds + str(0) + "/"
        try:
            self.hist = {
                key: self.hdf_file[self.hdf_key_norm + key][:]
                for key in self.hdf_keys_ds_hist
            }
        except KeyError:
            # load downsampled histograms if regular histograms don't work
            self.hist = {
                key: self.hdf_file[pyramid_level + key][:]
                for key in self.hdf_keys_ds_hist
            }
            for key in self.hdf_keys_ds_hist_scalar:
                self.hist[key] = self.hdf_file[pyramid_level + key][()]

        ds_data_key = pyramid_level + self.hdf_key_data
        self.data_ds = self.hdf_file[ds_data_key]

    @_check_and_open_hdf
    def _unload_hdf_normalized_and_ds(self):
        self._data = self.hdf_file[self.hdf_key_norm_proj]
        self.data = self._data
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
    def _return_ds_data(self, pyramid_level=0, px_range=None):

        pyramid_level = self.hdf_key_ds + str(pyramid_level) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        if px_range is None:
            self.data_returned = self.hdf_file[ds_data_key][:]
        else:
            x = px_range[0]
            y = px_range[1]
            self.data_returned = self.hdf_file[ds_data_key]
            self.data_returned = copy.deepcopy(
                self.data_returned[:, y[0] : y[1], x[0] : x[1]]
            )

    @_check_and_open_hdf
    def _return_data(self, px_range=None):
        if px_range is None:
            self.data_returned = self.hdf_file[self.hdf_key_norm_proj][:]
        else:
            x = px_range[0]
            y = px_range[1]
            self.data_returned = self.hdf_file[self.hdf_key_norm_proj][
                :, y[0] : y[1], x[0] : x[1]
            ]

    @_check_and_open_hdf
    def _return_hist(self, pyramid_level=0):
        pyramid_level = self.hdf_key_ds + str(pyramid_level) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        self.hist_returned = {
            key: self.hdf_file[pyramid_level + key][:] for key in self.hdf_keys_ds_hist
        }
        for key in self.hdf_keys_ds_hist_scalar:
            self.hist_returned[key] = self.hdf_file[pyramid_level + key][()]

    @_check_and_open_hdf
    def _delete_downsampled_data(self):
        if self.hdf_key_ds in self.hdf_file:
            del self.hdf_file[self.hdf_key_ds]

    def _close_hdf_file(self):
        if self.hdf_file:
            self.hdf_file.close()

    def _np_hist(self):
        r = [np.min(self.data), np.max(self.data)]
        bins = 200 if self.data.size > 200 else self.data.size
        hist = np.histogram(self.data, range=r, bins=bins)
        percentile = np.percentile(self.data.flatten(), q=(0.5, 99.5))
        bin_edges = hist[1]
        return hist, r, bins, percentile

    def _dask_hist(self):
        r = [da.min(self.data), da.max(self.data)]
        bins = 200 if self.data.size > 200 else self.data.size
        hist = da.histogram(self.data, range=r, bins=bins)
        percentile = da.percentile(self.data.flatten(), q=(0.5, 99.5))
        bin_edges = hist[1]
        return hist, r, bins, percentile

    def _dask_bin_centers(self, grp, write=False, savedir=None):
        tmp_filepath = copy.copy(self.filepath)
        tmp_filedir = copy.copy(self.filedir)
        if savedir is None:
            self.filedir = self.import_savedir
        else:
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
        self.filedir = tmp_filedir
        return bin_centers

    def _dask_hist_and_save_data(self):
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

    def _np_hist_and_save_data(self):
        hist, r, bins, percentile = self._np_hist()
        grp = IOBase.hdf_key_norm + "/"
        # self._data = da.from_array(self.data)
        # self.data = self._data
        data_dict = {
            self.hdf_key_norm_proj: self.data,
            grp + self.hdf_key_bin_frequency: hist[0],
            grp + self.hdf_key_bin_edges: hist[1],
            grp + self.hdf_key_image_range: r,
            grp + self.hdf_key_percentile: percentile,
        }
        self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
        self._dask_bin_centers(grp, write=True, savedir=self.import_savedir)

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

    def save_normalized_as_tiff(self):
        """
        Saves current self.data under the current self.filedir as
        self.normalized_projections_tif_key
        """
        tf.imwrite(self.filedir / str(self.normalized_projections_tif_key), self.data)

    def dask_data_to_h5(self, data_dict, savedir=None):
        """
        Brings lazy dask arrays to hdf5.

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

    def make_import_savedir(self, folder_name):
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

    def copy_from_parent(self):
        # self.parent_projections._unload_hdf_normalized_and_ds()
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

    def get_parent_data_from_hdf(self, px_range=None):
        """
        Gets data from hdf file and stores it in self.data.
        Parameters
        ----------
        px_range: tuple
            tuple of two two-element lists - (px_range_x, px_range_y)
        """
        self.data_ds = None
        self.parent_projections._unload_hdf_normalized_and_ds()
        self.parent_projections._return_data(px_range)
        self._data = self.parent_projections.data_returned
        self.data = self._data
        self.parent_projections._close_hdf_file()

    def get_parent_data_ds_from_hdf(self, pyramid_level, px_range=None):
        self.parent_projections._unload_hdf_normalized_and_ds()
        self.parent_projections._return_ds_data(pyramid_level, px_range)
        self.data_ds = self.parent_projections.data_returned
        self.parent_projections._close_hdf_file()

    def get_parent_hists(self, pyramid_level):
        self.parent_projections._unload_hdf_normalized_and_ds()
        self.parent_projections._return_hist(pyramid_level)
        self.hist = self.parent_projections.hist_returned
        self.parent_projections._close_hdf_file()

    def shift_and_save_projections(self, sx, sy):
        self.get_parent_data_from_hdf()
        self.sx = [sx]
        self.sy = [sy]
        for i in range(3):
            self.sx.append([shift / (2 ** i) for shift in sx])
            self.sy.append([shift / (2 ** i) for shift in sy])
        self._data = shift_projections(
                self.data, self.sx, self.sy
            )
        self.data = self._data
        data_dict = {self.hdf_key_norm_proj: self.data}
        self.projections.dask_data_to_h5(data_dict)
        for i in range(3):
            self.get_parent_data_ds_from_hdf(i)
            self.sx

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
        if Uploader.imported_metadata and not self.tiff_folder:
            self.import_file_projections(Uploader)
            return
        self.tic = time.perf_counter()
        Uploader.import_status_label.value = "Importing file directory."
        self.filedir = Uploader.filedir
        if not Uploader.imported_metadata:
            self.make_import_savedir(str(self.metadata.metadata["energy_str"] + "eV"))
            self.metadata.set_attributes_from_metadata_before_import(self)
        cwd = os.getcwd()
        os.chdir(self.filedir)
        try:
            self._data = dask_image.imread.imread("*.tif").astype(np.float32)
        except:
            self._data = dask_image.imread.imread("*.tiff").astype(np.float32)
        self.data = self._data
        self.metadata.set_metadata_from_attributes_after_import(self)
        self.filedir = self.import_savedir
        self.save_data_and_metadata(Uploader)
        self._check_downsampled_data()
        os.chdir(cwd)
        self.filepath = self.import_savedir / self.normalized_projections_hdf_key
        self.hdf_file = h5py.File(self.filepath)
        self._close_hdf_file()

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
        if self.tiff_folder:
            self.import_filedir_projections(Uploader)
            return
        self.tic = time.perf_counter()
        Uploader.import_status_label.value = "Importing single file."
        self.imported = False
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        if self.filename is None or self.filename == "":
            self.filename = str(Uploader.images_in_dir[0].name)
        self.filepath = self.filedir / self.filename
        if not Uploader.imported_metadata:
            "trying to make importsavedir"
            self.make_import_savedir(str(self.metadata.metadata["energy_str"] + "eV"))
            self.metadata.set_attributes_from_metadata_before_import(self)
            self.filedir = self.import_savedir

        # if import metadata is found in the directory, self.normalized_projections_npy_key
        # will be uploaded. This behavior is probably OK if we stick to this file
        # structure
        if Uploader.imported_metadata:
            files = [
                pathlib.Path(f).name for f in os.scandir(self.filedir) if not f.is_dir()
            ]
            if self.normalized_projections_hdf_key in files:
                Uploader.import_status_label.value = (
                    "Detected metadata and hdf5 file in this directory,"
                    + " uploading normalized_projections.hdf5"
                )
            elif ".npy" in files:
                Uploader.import_status_label.value = (
                    "Detected metadata and npy file in this directory,"
                    + " uploading normalized_projections.npy"
                )
                self._data = np.load(
                    self.filedir / "normalized_projections.npy"
                ).astype(np.float32)
                self.data = self._data
            if Uploader.save_tiff_on_import_checkbox.value:
                Uploader.import_status_label.value = "Saving projections as .tiff."
                self.saved_as_tiff = True
                self.save_normalized_as_tiff()
                self.metadata.metadata["saved_as_tiff"] = True
            self.metadata.set_attributes_from_metadata(self)
            self._check_downsampled_data(label=Uploader.import_status_label)
            self.imported = True

        elif any([x in self.filename for x in [".tif", ".tiff"]]):
            self._data = dask_image.imread.imread(self.filepath).astype(np.float32)
            self._data = da.where(da.isfinite(self._data), self._data, 0)
            self.data = self._data
            self.save_data_and_metadata(Uploader)
            self.imported = True
            self.filepath = self.import_savedir / self.normalized_projections_hdf_key

        elif ".npy" in self.filename:
            self._data = np.load(self.filepath).astype(np.float32)
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self._data = da.from_array(self._data)
            self.data = self._data
            self.save_data_and_metadata(Uploader)
            self.imported = True
            self.filepath = self.import_savedir / self.normalized_projections_hdf_key

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

    def save_data_and_metadata(self, Uploader):
        """
        Saves current data and metadata in import_savedir.
        """
        self.filedir = self.import_savedir
        self._dask_hist_and_save_data()
        self.saved_as_tiff = False
        if Uploader.save_tiff_on_import_checkbox.value:
            Uploader.import_status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata.metadata["saved_as_tiff"] = True
        self.metadata.filedir = self.filedir
        self.toc = time.perf_counter()
        self.metadata.metadata["import_time"] = self.toc - self.tic
        self.metadata.set_metadata_from_attributes_after_import(self)
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

    def normalize_nf(self):
        """
        Wrapper for tomopy's normalize_nf
        """
        os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
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
        os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
        if int(self.flats.shape[1]) == int(2 * self._data.shape[1]):
            self.flats = ndi.zoom(self.flats,(1,0.5,0.5))
            self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
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

        # Don't know if putting @staticmethod above a decorator will mess it up, so this
        # fct is inside. This is kind of funky. TODO.
        @dask.delayed
        def divide_arrays(x, ind):
            y = denominator[ind]
            return np.true_divide(x, y)

        if len(flat_loc) != 1:
            proj_locations = [
                int(np.ceil((flat_loc[i] + flat_loc[i + 1]) / 2))
                for i in range(len(flat_loc) - 1)
            ]
            chunk_setup = [int(np.ceil(proj_locations[0]))]
            for i in range(len(proj_locations) - 1):
                chunk_setup.append(proj_locations[i + 1] - proj_locations[i])
            chunk_setup.append(projs.shape[0] - sum(chunk_setup))
            chunk_setup = tuple(chunk_setup)
            projs_rechunked = projs.rechunk(
                {0: chunk_setup, 1: -1, 2: -1}
            )  # chunk data
            projs_rechunked = projs_rechunked - dark
            if status_label is not None:
                status_label.value = f"Dividing by flatfields and taking -log."
            blocks = projs_rechunked.to_delayed().ravel()
            results = [
                da.from_delayed(
                    divide_arrays(b, i),
                    shape=(
                        chunksize,
                        projs_rechunked.shape[1],
                        projs_rechunked.shape[2],
                    ),
                    dtype=np.float32,
                )
                for i, (b, chunksize) in enumerate(zip(blocks, chunk_setup))
            ]
            arr = da.concatenate(results, axis=0, allow_unknown_chunksizes=True)
        else:
            # if only 1 set of flats was taken, just divide normally.
            arr = projs / flats_reduced

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


