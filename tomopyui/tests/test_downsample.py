from .test_io import hdf_manager
from tomopyui.backend.util.dask_downsample import pyramid_reduce_gaussian
from tomopyui.backend.hdf_manager import hdf_key_norm, hdf_key_data
import pathlib
import h5py
from dask.array import core as da_core
from numpy.testing import assert_array_almost_equal
import numpy as np


hfd_manager = hdf_manager


def test_downsample_no_manager(hdf_manager):
    filepath: pathlib.Path = hdf_manager.filepath
    before_ds: np.ndarray = np.array([])
    with hdf_manager("a") as manager:
        before_ds = manager.get_ds_data(pyramid_level=0, lazy=False)
        manager.delete_ds_data()
    with h5py.File(filepath, "a") as f:
        key = f"{hdf_key_norm}/{hdf_key_data}"
        # Perform downsampling or any other operation that modifies the data
        coarsened_images = pyramid_reduce_gaussian(
            image=da_core.from_array(f[key], chunks="auto"),
            pyramid_levels=2,
            compute=True,
        )
        if coarsened_images:
            coarsened_image = coarsened_images[0]

        # Use numpy's testing utilities to assert similarity within a tolerance
        assert_array_almost_equal(
            before_ds,
            coarsened_image,
            decimal=6,
            err_msg="Downsampled arrays are not sufficiently similar.",
        )


def test_downsampling_shape_similarity(hdf_manager):
    with hdf_manager("a") as manager:
        original_ds_shape = manager.get_ds_data(pyramid_level=0, lazy=False).shape
        original_ds_data = manager.get_ds_data(pyramid_level=0, lazy=False)
        pyramid_reduce_gaussian(
            hdf_manager=manager,
            pyramid_levels=2,
            compute=False,
        )
        # Assuming your downsampling routine updates the HDF file with the downsampled dataset
        ds_shape = manager.get_ds_data(pyramid_level=0, lazy=False).shape
        ds_data = manager.get_ds_data(pyramid_level=0, lazy=False)
        # Check if the downsampled shape is as expected
        assert_array_almost_equal(
            original_ds_data,
            ds_data,
            decimal=5,
            err_msg="Downsampled arrays are not sufficiently similar.",
        )
