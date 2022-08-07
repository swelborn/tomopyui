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
from skimage.util import img_as_float32
from skimage.exposure import rescale_intensity
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
        self.angles_from_filenames = True
        if self.scan_info["REFEVERYEXPOSURES"] == 1 and self.scan_type == "ENERGY_TOMO":
            (
                self.flats_filenames,
                self.data_filenames,
            ) = self.get_all_data_filenames_filedir(Uploader.filedir)
            self.angles_from_filenames = False
            self.from_txrm = True
            self.from_xrm = False
        else:
            (
                self.flats_filenames,
                self.data_filenames,
            ) = self.get_all_data_filenames()
            self.txrm = False
            self.from_xrm = True
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
            self.get_angles_from_txrm()
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
        if self.from_xrm:
            self.import_from_run_script(Uploader)
        elif self.from_txrm:
            self.import_from_txrm(Uploader)
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

    def get_all_data_filenames_filedir(self, filedir):
        """
        Grabs the flats and projections filenames from scan info.

        Returns
        -------
        flats: list of pathlib.Path
            All flat file names in self.scan_info["FILES"]
        projs: list of pathlib.Path
            All projection file names in self.scan_info["FILES"]
        """
        txrm_files = self._file_finder(filedir, [".txrm"])
        xrm_files = self._file_finder(filedir, [".xrm"])
        txrm_files = [filedir / file for file in txrm_files]
        xrm_files = [filedir / file for file in xrm_files]
        if any(["ref_" in str(file) for file in txrm_files]):
            flats = [
                file.parent / file.name for file in txrm_files if "ref_" in file.name
            ]
        else:
            flats = [
                file.parent / file.name for file in xrm_files if "ref_" in file.name
            ]
        if any(["tomo_" in str(file) for file in txrm_files]):
            projs = [
                file.parent / file.name for file in txrm_files if "tomo_" in file.name
            ]
        else:
            projs = [
                file.parent / file.name for file in xrm_files if "tomo_" in file.name
            ]
        return flats, projs

    def get_angles_from_filenames(self):
        """
        Grabs the angles from the file names in scan_info.
        """
        reg_exp = re.compile("_[+-0]\d\d\d.\d\d")
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

    def get_angles_from_txrm(self):
        """
        Gets the angles from the raw image metadata.
        """
        self.angles_rad = self.scan_info["PROJECTION_METADATA"][0]["thetas"]
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

    def load_txrm(self, txrm_filepath):
        data, metadata = read_txrm(str(txrm_filepath))
        # rescale -- camera saturates at 4k -- can double check this number later.
        # should not impact reconstruction
        data = rescale_intensity(data, in_range=(0,4096),out_range="dtype") 
        data = img_as_float32(data)
        return data, metadata

    def import_from_txrm(self, Uploader):
        """
        Script to upload selected data from selected txrm energies.

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
        parent_metadata = self.metadata.metadata.copy()
        if "data_hierarchy_level" not in parent_metadata:
            try:
                with open(self.filepath) as f:
                    parent_metadata = json.load(
                        self.run_script_path.parent / "raw_metadata.json"
                    )
            except Exception:
                pass
        for energy in self.selected_energies:
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
                display(self.energy_label)
            # Getting filename from specific energy
            self.flats_filename = [
                file.parent / file.name
                for file in self.flats_filenames
                if energy in file.name and "ref_" in file.name
            ]
            self.data_filename = [
                file.parent / file.name
                for file in self.data_filenames
                if energy in file.name and "tomo_" in file.name
            ]
            self.status_label = Label(
                "Uploading txrm.", layout=Layout(justify_content="center")
            )
            self.flats, self.scan_info["FLAT_METADATA"] = self.load_txrm(
                self.flats_filename[0]
            )
            self._data, self.scan_info["PROJECTION_METADATA"] = self.load_txrm(
                self.data_filename[0]
            )
            self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
            self.make_import_savedir(str(energy + "eV"))
            self.status_label.value = "Normalizing."
            self.normalize()
            self._data = np.flip(self._data, axis=1)
            #TODO: potentially do this in normalize, decide later
            # this removes negative values, 
            self._data = self._data - np.median(self._data[self._data < 0])
            self._data[self._data < 0] = 0.0
            self.data = self._data
            self.status_label.value = "Calculating histogram of raw data and saving."
            self._np_hist_and_save_data()
            self.saved_as_tiff = False
            self.filedir = self.import_savedir
            if Uploader.save_tiff_on_import_checkbox.value:
                self.status_label.value = "Saving projections as .tiff."
                self.saved_as_tiff = True
                self.save_normalized_as_tiff()
            self.status_label.value = "Downsampling data."
            self._check_downsampled_data()
            self.status_label.value = "Saving metadata."
            self.data_hierarchy_level = 1
            self.metadata.set_metadata(self)
            self.metadata.filedir = self.import_savedir
            self.metadata.filename = "import_metadata.json"
            self.metadata.save_metadata()
            self.filedir = _tmp_filedir
            self._close_hdf_file()

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
                self.make_import_savedir(str(energy + "eV"))
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
                self.status_label.value = (
                    "Calculating histogram of raw data and saving."
                )
                self._dask_hist_and_save_data()
                self.saved_as_tiff = False
                self.filedir = self.import_savedir
                if Uploader.save_tiff_on_import_checkbox.value:
                    self.status_label.value = "Saving projections as .tiff."
                    self.saved_as_tiff = True
                    self.save_normalized_as_tiff()
                self.status_label.value = "Downsampling data."
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
        self.import_metadata()
        self.metadata_projections.set_extra_metadata(Uploader)
        self.metadata_references.set_extra_metadata(Uploader)
        self.metadata.filedir = self.metadata_projections.filedir
        self.filedir = self.metadata.filedir
        self.metadata.filepath = self.metadata.filedir / self.metadata.filename
        self.metadata.save_metadata()
        save_filedir_name = str(self.metadata_projections.metadata["energy_str"] + "eV")
        self.import_savedir = self.metadata_projections.filedir / save_filedir_name
        self.make_import_savedir(save_filedir_name)
        self.import_filedir_projections(Uploader)
        self.import_filedir_flats(Uploader)
        self.filedir = self.import_savedir
        projs, flats, darks = self.setup_normalize(Uploader)
        Uploader.import_status_label.value = "Normalizing projections"
        self._data = self.normalize_no_locations_no_average(
            projs, flats, darks, compute=False
        )
        self.data = self._data
        hist, r, bins, percentile = self._dask_hist()
        grp = self.hdf_key_norm
        data_dict = {
            self.hdf_key_norm_proj: self.data,
            grp + self.hdf_key_bin_frequency: hist[0],
            grp + self.hdf_key_bin_edges: hist[1],
            grp + self.hdf_key_image_range: r,
            grp + self.hdf_key_percentile: percentile,
        }
        self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
        self._dask_bin_centers(grp, write=True, savedir=self.import_savedir)
        Uploader.import_status_label.value = "Downsampling data in a pyramid"
        self.filedir = self.import_savedir
        self._check_downsampled_data(label=Uploader.import_status_label)
        self.metadata_projections.set_attributes_from_metadata(self)
        self.metadata_prenorm = Metadata_SSRL62B_Prenorm()
        self.metadata_prenorm.set_metadata(self)
        self.metadata_prenorm.metadata[
            "parent_metadata"
        ] = self.metadata.metadata.copy()
        if Uploader.save_tiff_on_import_checkbox.value:
            self.status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata["saved_as_tiff"] = projections.saved_as_tiff
        self.metadata_prenorm.filedir = self.filedir
        self.metadata_prenorm.filepath = self.filedir / self.metadata_prenorm.filename
        self.metadata_prenorm.save_metadata()

        self.hdf_file.close()

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
        data_dict = {self.hdf_key_raw_proj: arr}
        da.to_hdf5(self.import_savedir / self.normalized_projections_hdf_key, data_dict)

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
        data_dict = {self.hdf_key_raw_flats: arr}
        da.to_hdf5(self.import_savedir / self.normalized_projections_hdf_key, data_dict)

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
        self.hdf_file = h5py.File(
            self.import_savedir / self.normalized_projections_hdf_key, "a"
        )
        self.flats = self.hdf_file[self.hdf_key_raw_flats]
        self._data = self.hdf_file[self.hdf_key_raw_proj]
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
        self.data = da.from_array(self.data, chunks={0: "auto", 1: -1, 2: -1})
        self.import_status_label.value = "Saving projections as hdf"
        self.save_data_and_metadata(Uploader)

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

    def save_data_and_metadata(self, Uploader):
        """
        Saves current data and metadata in import_savedir.
        """
        self.filedir = self.import_savedir
        self._dask_hist_and_save_data()
        self.saved_as_tiff = False
        _metadata = self.metadata.metadata.copy()
        if Uploader.save_tiff_on_import_checkbox.value:
            Uploader.import_status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata.metadata["saved_as_tiff"] = True
        self.metadata.filedir = self.filedir
        self.toc = time.perf_counter()
        self.metadata = self.save_normalized_metadata(self.toc - self.tic, _metadata)
        Uploader.import_status_label.value = "Checking for downsampled data."
        self._check_downsampled_data(label=Uploader.import_status_label)


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