import os
import pathlib
import numpy as np
import dask.array.reductions as da_red
import dask.array.routines as da_rout
import dask.array.core as da_core
import datetime
from typing import Optional, Union
import pathlib
import json
import numpy as np
import h5py
import tifffile as tf
from typing import List, Tuple, Callable, Dict, Any
import pathlib
import json
import numpy as np
import h5py
import tifffile as tf
from ipywidgets import Checkbox
from typing import Tuple, Any, Optional
import pathlib
import json
import tifffile as tf
import dask.array.percentile as da_perc
from tomopy.sim.project import angles as angle_maker


# Define a type for the size extraction functions
SizeExtractor = Callable[[pathlib.Path, Any], Tuple[int, int, int]]


def extract_tiff_size(image: pathlib.Path, _: Any) -> Tuple[int, int, int]:
    try:
        with tf.TiffFile(image) as tif:
            # Attempt to access the ImageDescription tag if present
            image_description = tif.pages[0].tags.get("ImageDescription", None)
            if image_description:
                size = json.loads(image_description.value)["shape"]
                return size[0], size[1], size[2]
            else:
                # Fallback to basic TIFF dimensions if ImageDescription is not usable
                return tif.pages[0].shape[0], tif.pages[0].shape[1], len(tif.pages)
    except Exception:
        return 1, 1, 1


def extract_npy_size(image: pathlib.Path, _: Any) -> Tuple[int, int, int]:
    try:
        size = np.load(image, mmap_mode="r").shape
        return size[0], size[1], size[2]
    except Exception:
        return 1, 1, 1


def extract_hdf5_size(image: pathlib.Path, projections: Any) -> Tuple[int, int, int]:
    try:
        with h5py.File(image, "r") as f:
            dataset = f.get(projections.hdf_key_norm_proj)
            if isinstance(dataset, h5py.Dataset):
                return dataset.shape[0], dataset.shape[1], dataset.shape[2]
            else:
                raise ValueError("Dataset key not found")
    except Exception:
        return 1, 1, 1


def extract_image_sizes(
    image_list: List[pathlib.Path], tiff_folder_checkbox: Checkbox, projections: Any
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Extracts the sizes of images in a list and updates the tiff_folder_checkbox based on the count of TIFF images.

    Parameters:
    - image_list: List of pathlib.Path objects representing image files.
    - tiff_folder_checkbox: A Checkbox widget to be updated based on the TIFF images count.
    - projections: An object for handling HDF5 files, containing attributes like `filepath` and `hdf_key_norm_proj`.

    Returns:
    - A list of tuples, each representing the size (Z, Y, X) of an image.
    - The count of TIFF images in the folder.
    """
    size_list = []
    tiff_count_in_folder = len(
        [file for file in image_list if file.suffix in [".tiff", ".tif"]]
    )

    # Update the tiff_folder_checkbox based on TIFF count
    tiff_folder_checkbox.disabled = tiff_count_in_folder <= 1
    tiff_folder_checkbox.value = tiff_count_in_folder > 1

    # Mapping from file extension to the corresponding size extractor function
    size_extractors: Dict[str, SizeExtractor] = {
        ".tiff": extract_tiff_size,
        ".tif": extract_tiff_size,
        ".npy": extract_npy_size,
        ".hdf5": lambda image, proj: extract_hdf5_size(image, proj),
        ".h5": lambda image, proj: extract_hdf5_size(image, proj),
    }

    for image in image_list:
        extractor = size_extractors.get(image.suffix)
        if extractor:
            size_list.append(extractor(image, projections))
        else:
            size_list.append((1, 1, 1))  # Default size for unsupported file types

    return size_list, tiff_count_in_folder


def write_as_tiff(filepath: pathlib.Path, images: np.ndarray):
    tf.imwrite(filepath, images)


def file_finder(filedir: pathlib.Path, filetypes: list) -> Optional[list[str]]:
    """
    Used to find files of a given filetype in a directory.

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
    if not files_with_ext:
        return None
    return files_with_ext


def file_finder_fullpath(filedir, filetypes: list) -> list[pathlib.Path]:
    """
    Used to find files of a given filetype in a directory.

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


def make_timestamp_subdir(dir: pathlib.Path, suffix: str) -> pathlib.Path:
    time_formats = ["%Y%m%d-%H%M-", "%Y%m%d-%H%M%S-"]
    subdir = None
    for time_format in time_formats:
        now = datetime.datetime.now()
        dt_str = now.strftime(time_format)
        folder_name = dt_str + suffix
        subdir = dir / folder_name
        if not subdir.exists():
            subdir.mkdir()
            break
    else:
        raise FileExistsError(
            f"Unable to create a unique directory with the provided suffix: {suffix}"
        )
    return subdir


NpHistOutput = tuple[
    tuple[np.ndarray, np.ndarray], tuple[float, float], int, np.ndarray, np.ndarray
]


def np_hist(
    data: np.ndarray,
) -> NpHistOutput:
    r = (np.min(data), np.max(data))
    bins = 200 if data.size > 200 else data.size
    hist = np.histogram(data, range=r, bins=bins)
    percentile = np.percentile(data.flatten(), q=(0.5, 99.5)).astype(np.float32)
    centers = np_get_bin_centers_from_edges(hist[1])
    return hist, r, bins, centers, percentile


DaskHistOutput = tuple[
    tuple[da_core.Array, da_core.Array],
    da_core.Array,
    int,
    da_core.Array,
    da_core.Array,
]


def dask_hist(
    data: da_core.Array,
) -> DaskHistOutput:
    r_min = da_red.min(data)
    r_max = da_red.max(data)
    bins = 200 if data.size > 200 else data.size
    hist = da_rout.histogram(data, range=(r_min, r_max), bins=bins)
    percentile = da_perc(data.flatten(), q=(0.5, 99.5))
    centers = dask_get_bin_centers_from_edges(hist[1])
    image_range = da_core.stack([r_min, r_max])
    return hist, image_range, bins, centers, percentile


def hist(
    data: Union[np.ndarray, da_core.Array],
) -> Union[NpHistOutput, DaskHistOutput]:
    if isinstance(data, da_core.Array):
        return dask_hist(data)
    elif isinstance(data, np.ndarray):
        return np_hist(data)


def np_get_bin_centers_from_edges(bin_edges: np.ndarray) -> np.ndarray:
    edges_left = bin_edges[:-1]
    edges_right = bin_edges[1:]
    centers = (edges_left + edges_right) / 2
    return centers


def dask_get_bin_centers_from_edges(bin_edges: da_core.Array) -> da_core.Array:
    # Calculate bin centers in a way that leverages Dask's capabilities
    edges_left = bin_edges[:-1]
    edges_right = bin_edges[1:]
    centers = (edges_left + edges_right) / 2
    return centers


def make_angles(
    pxZ, angle_start: float, angle_end: float
) -> tuple[np.ndarray, np.ndarray]:
    angles_rad = angle_maker(
        pxZ,
        ang1=angle_start,
        ang2=angle_end,
    )
    angles_deg = np.ndarray([x * 180 / np.pi for x in angles_rad])
    return tuple(angles_rad, angles_deg)
