import dask.array.core as da_core
import numpy as np
import pathlib
import pytest
import shutil
import tempfile
from tomopyui.backend.hdf_manager import (
    HDFManager,
    hdf_key_bin_frequency,
    hdf_key_norm,
    hdf_key_bin_edges,
    hdf_key_image_range,
    hdf_key_percentile,
    hdf_key_bin_centers,
)
from tomopyui.backend.helpers import (
    np_hist,
    dask_hist,
    make_timestamp_subdir,
    file_finder,
    file_finder_fullpath,
)
import datetime

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
TEST_FILE = TEST_DATA_DIR / "normalized_projections.hdf5"


@pytest.fixture
def hdf_manager():
    temp_dir = tempfile.mkdtemp()
    temp_file = pathlib.Path(temp_dir) / TEST_FILE.name
    shutil.copy(TEST_FILE, temp_file)
    manager = HDFManager(temp_file)
    yield manager
    shutil.rmtree(temp_dir)


def test_get_normalized_data_lazy_and_non_lazy(hdf_manager: HDFManager):
    # Test non-lazy loading
    with hdf_manager("r"):
        np_data = hdf_manager.get_normalized_data(lazy=False)
        assert np_data is not None
        assert isinstance(
            np_data, np.ndarray
        ), "Non-lazy loading should return a NumPy array"

    # Test lazy loading
    with hdf_manager("r"):
        dask_data = hdf_manager.get_normalized_data(lazy=True)
        assert dask_data is not None
        assert isinstance(
            dask_data, da_core.Array
        ), "Lazy loading should return a Dask array"
        assert (
            dask_data.compute().all() == np_data.all()
        ), "Lazy and non-lazy loading should return the same data"


@pytest.mark.parametrize("pyramid_level", [0, 1, 2])
def test_get_ds_data_lazy_and_non_lazy(hdf_manager: HDFManager, pyramid_level: int):
    # Test non-lazy loading
    with hdf_manager:
        np_data = hdf_manager.get_ds_data(pyramid_level, lazy=False)
        assert np_data is not None
        assert isinstance(
            np_data, np.ndarray
        ), "Non-lazy loading should return a NumPy array"

    # Test lazy loading
    with hdf_manager:
        dask_data = hdf_manager.get_ds_data(pyramid_level, lazy=True)
        assert dask_data is not None
        assert isinstance(
            dask_data, da_core.Array
        ), "Lazy loading should return a Dask array"
        assert (
            dask_data.compute().all() == np_data.all()
        ), "Lazy and non-lazy loading should return the same data"


def test_delete_ds_data(hdf_manager: HDFManager):
    with hdf_manager("r+"):
        hdf_manager.get_ds_data(0, lazy=False)
        hdf_manager.delete_ds_data()
        assert hdf_manager.get_ds_data() is None


def test_save_hist(hdf_manager: HDFManager):
    with hdf_manager("r+") as manager:
        data = manager.get_normalized_data(lazy=False)
        assert data is not None
        assert isinstance(data, np.ndarray)
        hist_output = np_hist(data)

        assert data is not None
        assert hist_output is not None
        manager.save_hist(hist_output)

        assert manager.file is not None
        assert hdf_key_norm + hdf_key_bin_frequency in manager.file
        assert hdf_key_norm + hdf_key_bin_edges in manager.file
        assert hdf_key_norm + hdf_key_image_range in manager.file
        assert hdf_key_norm + hdf_key_percentile in manager.file
        assert hdf_key_norm + hdf_key_bin_centers in manager.file

    with hdf_manager("r+") as manager:
        data = manager.get_normalized_data(lazy=True)
        assert data is not None
        assert isinstance(data, da_core.Array)
        data = da_core.from_array(data.compute(), chunks="auto")
        assert isinstance(data, da_core.Array)
        hist_output = dask_hist(data)
        
        manager.save_hist(hist_output)

        assert manager.file is not None

        assert hdf_key_norm + hdf_key_bin_frequency in manager.file
        assert hdf_key_norm + hdf_key_bin_edges in manager.file
        assert hdf_key_norm + hdf_key_image_range in manager.file
        assert hdf_key_norm + hdf_key_percentile in manager.file
        assert hdf_key_norm + hdf_key_bin_centers in manager.file

    with hdf_manager as manager:
        data = manager.get_ds_data(1, lazy=False)
        assert data is not None
        hist_output = np_hist(data)
        manager.save_hist(hist_output)

        assert manager.file is not None
        assert hdf_key_norm + hdf_key_bin_frequency in manager.file
        assert hdf_key_norm + hdf_key_bin_edges in manager.file
        assert hdf_key_norm + hdf_key_image_range in manager.file
        assert hdf_key_norm + hdf_key_percentile in manager.file
        assert hdf_key_norm + hdf_key_bin_centers in manager.file


def test_delete_norm_hist(hdf_manager: HDFManager):
    with hdf_manager("r+"):
        deleted = hdf_manager.delete_norm_hist()
        assert deleted, "Histogram data should have been deleted"

    with hdf_manager:
        assert hdf_manager.file is not None
        for key in [
            hdf_key_norm + hdf_key_bin_frequency,
            hdf_key_norm + hdf_key_bin_edges,
            hdf_key_norm + hdf_key_image_range,
            hdf_key_norm + hdf_key_percentile,
            hdf_key_norm + hdf_key_bin_centers,
        ]:
            assert (
                key not in hdf_manager.file
            ), f"{key} should have been deleted from the HDF5 file"


def test_check_for_downsampled_data(hdf_manager):
    with hdf_manager:
        assert hdf_manager.check_for_downsampled_data()


def test_make_timestamp_subdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        parent_dir = pathlib.Path(tmpdir)
        suffix = "test-suffix"

        # Test creating the first directory
        subdir = make_timestamp_subdir(parent_dir, suffix)
        assert subdir.exists(), "Subdirectory should be created"

        # Verify the name format
        now = datetime.datetime.now()
        expected_name_starts = [
            now.strftime("%Y%m%d-%H%M-"),
            now.strftime("%Y%m%d-%H%M%S-"),
        ]
        assert any(
            subdir.name.startswith(prefix) for prefix in expected_name_starts
        ), "Subdirectory name should start with the expected timestamp format"

        # Test attempting to create another directory with the same suffix immediately
        # This should trigger the use of a more precise timestamp to avoid name collision
        subdir2 = make_timestamp_subdir(parent_dir, suffix)
        assert subdir2.exists(), "Second subdirectory should be created"
        assert (
            subdir != subdir2
        ), "Second subdirectory should have a different name to avoid collision"


@pytest.fixture
def temp_dir_with_files():
    # Setup temporary directory and files for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = pathlib.Path(tmpdir)
        # Create test files
        (dir_path / "test.txt").touch()
        (dir_path / "example.npy").touch()
        (dir_path / "ignore.me").touch()
        yield dir_path


def test_file_finder(temp_dir_with_files):
    expected_files = ["test.txt", "example.npy"]
    found_files = file_finder(temp_dir_with_files, [".txt", ".npy"])
    assert found_files
    assert sorted(found_files) == sorted(
        expected_files
    ), "file_finder did not return expected files"


def test_file_finder_fullpath(temp_dir_with_files):
    expected_files = [
        temp_dir_with_files / "test.txt",
        temp_dir_with_files / "example.npy",
    ]
    found_files = file_finder_fullpath(temp_dir_with_files, [".txt", ".npy"])
    assert sorted(found_files) == sorted(
        expected_files
    ), "file_finder_fullpath did not return expected file paths"
