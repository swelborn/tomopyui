import pathlib
import pytest
from tomopyui.backend.helpers import (
    extract_tiff_size,
    extract_hdf5_size,
    extract_image_sizes,
)
from unittest.mock import Mock
from ipywidgets import Checkbox


@pytest.fixture
def test_data_dir():
    # Adjust the path based on where this test file is located
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture
def tiff_file(test_data_dir):
    return test_data_dir / "tiff.tif"


@pytest.fixture
def tiff_folder(test_data_dir):
    return test_data_dir / "tiff_folder"


@pytest.fixture
def hdf5_file(test_data_dir):
    return test_data_dir / "normalized_projections.hdf5"


@pytest.fixture
def image_list(tiff_file, hdf5_file):
    return [tiff_file, hdf5_file]


@pytest.fixture
def projections():
    mock_projections = Mock()
    mock_projections.hdf_key_norm_proj = (
        "some_dataset_key"  # Adjust based on actual key in HDF5
    )
    return mock_projections


def test_extract_tiff_size(tiff_file):
    z, y, x = extract_tiff_size(tiff_file, None)
    assert (
        z > 0 and y > 0 and x > 0
    ), "Extracted size should be greater than 0 for each dimension"


def test_extract_hdf5_size(hdf5_file, projections):
    z, y, x = extract_hdf5_size(hdf5_file, projections)
    assert (
        z > 0 and y > 0 and x > 0
    ), "Extracted size should be greater than 0 for each dimension"


def test_extract_image_sizes(image_list, projections):
    tiff_folder_checkbox = Checkbox()
    sizes, tiff_count = extract_image_sizes(
        image_list, tiff_folder_checkbox, projections
    )
    assert len(sizes) == len(
        image_list
    ), "Should return a size tuple for each image in the list"
    assert (
        tiff_count > 0
    ), "TIFF count should be positive, indicating TIFF files were found"
