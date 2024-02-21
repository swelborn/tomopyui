import copy
import os
import pathlib
import multiprocessing
import time
from abc import ABC, abstractmethod

import dask
import dask.delayed
import dask.array as da
from dask.array import ufunc as da_ufunc
from dask.array import core as da_core
import copy
import dask_image.imread
import numpy as np
import scipy.ndimage as ndi
import tomopy.prep.normalize as tomopy_normalize
from ipywidgets import *
from typing import Optional
from tomopyui.backend.helpers import hist, make_timestamp_subdir
from tomopyui.backend.meta import Metadata
from tomopyui.backend.hdf_manager import HDFManager

os.environ["num_cpu_cores"] = str(multiprocessing.cpu_count())


class ProjectionsBase(ABC):
    """
    Base class for projections data. Abstract methods include importing/exporting data
    and metadata. One can import a file directory, or a particular file within a
    directory. Aliases give options for users to extract their data with other keywords.

    """

    def __init__(self):
        self.px_size: float = 1.0
        self.angles_rad: Optional[np.ndarray] = None
        self.angles_deg: Optional[np.ndarray] = None
        self.tiff_folder: bool = False

        self._data: np.ndarray = np.random.rand(10, 100, 100)
        self.data_ds: np.ndarray = self.data

        self.imported: bool = False
        self._filepath: pathlib.Path = pathlib.Path()
        self.filedir: pathlib.Path = pathlib.Path()
        self.filename: Optional[str] = None
        self.extension: Optional[str] = None
        self.import_savedir: pathlib.Path = pathlib.Path()

        self.shape: Optional[tuple] = None
        self.pxX: int = self._data.shape[2]
        self.pxY: int = self._data.shape[1]
        self.pxZ: int = self._data.shape[0]
        self.size_gb: Optional[float] = None
        self.parent: Optional[ProjectionsBase] = None
        self.energy: Optional[float] = None
        self.hist: Optional[np.ndarray] = None
        self.metadata: Optional[Metadata] = None
        self.hdf_manager: Optional[HDFManager] = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        (self.pxZ, self.pxY, self.pxX) = value.shape
        self.size_gb = value.nbytes / 1048576 / 1000  # Convert bytes to gigabytes

    @property
    def filepath(self) -> pathlib.Path:
        return self._filepath

    @filepath.setter
    def filepath(self, value: pathlib.Path):
        self._filepath = value
        self.filedir = value.parent
        self.filename = value.name
        self.extension = value.suffix

    def deepcopy_from_parent(self):
        if self.parent:
            new: ProjectionsBase = copy.deepcopy(self.parent)
            new.parent = self.parent
            self = new

    # def get_parent_data_from_hdf(self, px_range=None):
    #     """
    #     Gets data from hdf file and stores it in self.data.
    #     Parameters
    #     ----------
    #     px_range: tuple
    #         tuple of two two-element lists - (px_range_x, px_range_y)
    #     """
    #     self.data_ds = None
    #     self.parent._unload_hdf_normalized_and_ds()
    #     self.parent._return_data(px_range)
    #     self._data = self.parent.data_returned
    #     self.data = self._data
    #     self.parent._close_hdf_file()

    # def get_parent_data_ds_from_hdf(self, pyramid_level, px_range=None):
    #     self.parent._unload_hdf_normalized_and_ds()
    #     self.parent._return_ds_data(pyramid_level, px_range)
    #     self.data_ds = self.parent.data_returned
    #     self.parent._close_hdf_file()

    # @abstractmethod
    # def import_metadata(self, filedir): ...

    # @abstractmethod
    # def import_filedir_projections(self, filedir): ...

    # @abstractmethod
    # def import_file_projections(self, filepath): ...


class Projections_Prenormalized(ProjectionsBase):
    """ """

    def import_tiff(
        self, filedir: pathlib.Path, status_label: Optional[widgets.Label] = None
    ) -> bool:
        """
        Similar process to import_file_projections. This one will be triggered if the
        tiff folder checkbox is selected on the frontend.
        """
        tic = time.perf_counter()

        if status_label:
            status_label.value = "Importing file directory."

        self.filedir = filedir

        tif_files = list(self.filedir.glob("*.tif"))
        tiff_files = list(self.filedir.glob("*.tiff"))

        if tif_files and tiff_files:
            if status_label:
                status_label.value = "Found both .tif and .tiff files in this directory. Please just use one file type."
                return False

        tif_files = tif_files + tiff_files

        if len(tif_files) > 1:
            _import_tiffstack(tif_files, status_label)
        else:
            _import_tiff(tif_files[0], status_label)

        # TODO: ENERGY_STR
        self.filedir = make_timestamp_subdir(self.filedir, ENERGY_STR)
        _hist = hist(self._data)

        self.hdf_manager = HDFManager(self.filedir / "normalized_projections.hdf5")

        # save histogram and data first, then downsample
        with self.hdf_manager("r+"):
            self.hdf_manager.save_normalized_data(self._data)
            self.hdf_manager.save_hist(_hist)

        self.saved_as_tiff = False
        # if Uploader.save_tiff_on_import_checkbox.value:
        #     Uploader.import_status_label.value = "Saving projections as .tiff."
        #     self.saved_as_tiff = True
        #     self.save_normalized_as_tiff()
        #     self.metadata.metadata["saved_as_tiff"] = True
        self.metadata.filedir = self.filedir
        toc = time.perf_counter()
        import_time = toc - tic

        self.metadata.metadata["import_time"] = toc - tic
        self.metadata.set_metadata_from_attributes_after_import(self)
        self.metadata.save_metadata()
        if status_label:
            status_label.value = "Downsampling data (if not already)."

        self._check_downsampled_data(label=Uploader.import_status_label)

        self.data = self._data
        # self.metadata.set_metadata_from_attributes_after_import(self)
        self.filedir = self.import_savedir
        self.save_data_and_metadata(Uploader)
        self._check_downsampled_data()

        self.filepath = self.import_savedir / self.normalized_projections_hdf_key

        return True

    def import_npy(
        self, filepath: pathlib.Path, status_label: Optional[widgets.Label] = None
    ):
        pass

    def import_hdf5(
        self, filepath: pathlib.Path, status_label: Optional[widgets.Label] = None
    ):
        pass

    def _import_tiff(
        self, filepath: pathlib.Path, status_label: Optional[widgets.Label] = None
    ):

        if status_label:
            status_label.value = "Importing single tiff."

        self._data = dask_image.imread.imread(filepath).astype(np.float32)
        pass

    def _import_tiffstack(
        self, paths: list[pathlib.Path], status_label: Optional[widgets.Label] = None
    ):

        if status_label:
            status_label.value = "Importing multiple tiffs."

        self.tic = time.perf_counter()
        self.imported = False
        self.filedir = filepath.parent
        self.filename = filepath.name
        self.filepath = filepath

        tif_files = list(self.filedir.glob("*.tif"))
        tiff_files = list(self.filedir.glob("*.tiff"))

        self._data = dask_image.imread.imread(paths).astype(np.float32)
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

    def import_metadata(self, filepath):
        self.metadata = Metadata.parse_metadata_type(filepath)
        self.metadata.load_metadata()


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
        self.flats: Optional[np.ndarray] = None
        self.flats_ind: Optional[list] = None
        self.darks: Optional[np.ndarray] = None
        self.normalized: bool = False

    def normalize_nf(self):
        """
        Wrapper for tomopy's normalize_nf
        """
        os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
        self._data = tomopy_normalize.normalize_nf(
            self._data, self.flats, self.darks, self.flats_ind
        )
        self._data: np.ndarray = tomopy_normalize.minus_log(self._data)
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
            self.flats = ndi.zoom(self.flats, (1, 0.5, 0.5))
            self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
        self._data = tomopy_normalize.normalize(self._data, self.flats, self.darks)
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    # For dask-based normalization (should be faster)
    @staticmethod
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
            da_core.from_delayed(
                mean_on_chunks(b),
                shape=(1, chunked_da.shape[1], chunked_da.shape[2]),
                dtype=np.float32,
            )
            for b in blocks
        ]
        # arr not computed yet
        arr = da_core.concatenate(results, axis=0, allow_unknown_chunksizes=True)
        return arr

    @staticmethod
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
                da_core.from_delayed(
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
            arr = da_core.concatenate(results, axis=0, allow_unknown_chunksizes=True)
        else:
            # if only 1 set of flats was taken, just divide normally.
            arr = projs / flats_reduced

        arr = arr.rechunk((num_exposures_per_proj, -1, -1))
        arr = RawProjectionsBase.average_chunks(arr).astype(np.float32)
        arr = -da_ufunc.log(arr)
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
        projs = -da_ufunc.log(projs)
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
