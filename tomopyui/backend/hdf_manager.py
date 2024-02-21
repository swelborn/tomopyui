import pathlib

import h5py
from typing import Optional, Any, Union
import dask.array.core as da_core
import numpy as np

from tomopyui.backend.helpers import DaskHistOutput, NpHistOutput

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


def ensure_file_open(func):
    def wrapper(self, *args, **kwargs):
        if self.file is None:
            raise ValueError(
                "HDF5 file is not open. Use the context manager to open it."
            )
        return func(self, *args, **kwargs)

    return wrapper


class HDFManager:

    def __init__(self, filepath: pathlib.Path):
        self.filepath: pathlib.Path = filepath
        self.file: Optional[h5py.File] = None
        self.mode: str = "r"

    def __call__(self, mode: str):
        self.mode = mode
        return self

    def __enter__(self):
        self._open_file(self.mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        self.file = None

    def _open_file_read_only(self):
        self._open_file("r")

    def _open_file_read_write(self):
        self._open_file("r+")

    def _open_file_append(self):
        self._open_file("a")

    def _open_file(self, mode: str = "r"):
        if self.file:
            self.file.close()  # Close any previously opened file first
        self.mode = mode
        self.file = h5py.File(self.filepath, mode)

    def set_filepath(self, filepath: pathlib.Path):
        self.filepath = filepath
        if self.file:
            self.file.close()
            self.file = None

    @ensure_file_open
    def get_normalized_data(
        self, lazy: bool = False
    ) -> Optional[Union[np.ndarray, da_core.Array]]:
        assert self.file is not None
        dataset = self.file.get(f"{hdf_key_norm_proj}")
        if dataset is None or not isinstance(dataset, h5py.Dataset):
            print(
                f"Expected a Dataset but got {type(dataset)} for key '{hdf_key_norm_proj}'."
            )
            return None

        if lazy:
            return da_core.from_array(dataset, chunks="auto")
        else:
            return dataset[:]

    @ensure_file_open
    def get_hist(self, pyramid_level: int = 0) -> Optional[dict[str, Any]]:
        """Get histogram data for a given pyramid level."""
        if self.file is None:
            raise ValueError(
                "HDF5 file is not open. Use the context manager to open it."
            )
        hist_keys = hdf_keys_ds_hist
        hist = {}
        hdf_key_prefix = f"{hdf_key_ds}{pyramid_level}/"

        for key in hist_keys:
            dataset = self.file.get(hdf_key_prefix + key)
            if dataset is not None and isinstance(dataset, h5py.Dataset):
                hist[key] = dataset[:]
            else:
                print(
                    f"Expected a Dataset but got {type(dataset)} for key '{hdf_key_prefix + key}'."
                )
                return None
        return hist

    @ensure_file_open
    def get_ds_data(self, pyramid_level: int = 0, lazy: bool = False) -> Optional[Any]:
        if self.file is None:
            raise ValueError(
                "HDF5 file is not open. Use the context manager to open it."
            )
        hdf_key = f"{hdf_key_ds}{pyramid_level}/{hdf_key_data}"
        dataset = self.file.get(hdf_key)

        if dataset is None or not isinstance(dataset, h5py.Dataset):
            print(f"Expected a Dataset but got {type(dataset)} for key '{hdf_key}'.")
            return None

        if lazy:
            return da_core.from_array(dataset, chunks="auto")
        else:
            return dataset[:]  # Return as NumPy array

    @ensure_file_open
    def delete_ds_data(self) -> bool:
        assert self.file is not None
        if hdf_key_ds in self.file:
            del self.file[f"{hdf_key_ds}"]
            print("Downsampled data deleted successfully.")
            return True
        else:
            print("No downsampled data found to delete.")
            return False

    @ensure_file_open
    def delete_ds_hist(self) -> bool:
        hist_keys = [
            hdf_key_ds + hdf_key_bin_frequency,
            hdf_key_ds + hdf_key_bin_edges,
            hdf_key_ds + hdf_key_image_range,
            hdf_key_ds + hdf_key_percentile,
            hdf_key_ds + hdf_key_bin_centers,
        ]
        return self.delete_hist(hist_keys)

    @ensure_file_open
    def delete_norm_hist(self) -> bool:
        # List of histogram-related keys to be deleted
        hist_keys = [
            hdf_key_norm + hdf_key_bin_frequency,
            hdf_key_norm + hdf_key_bin_edges,
            hdf_key_norm + hdf_key_image_range,
            hdf_key_norm + hdf_key_percentile,
            hdf_key_norm + hdf_key_bin_centers,
        ]
        return self.delete_hist(hist_keys)

    @ensure_file_open
    def delete_hist(self, keys_to_delete: list[str]) -> bool:
        # List of histogram-related keys to be deleted

        deleted = False
        assert self.file is not None
        for key in keys_to_delete:
            if key in self.file:
                del self.file[key]
                deleted = True
            else:
                print(f"{key} not found in HDF5 file.")

        if not deleted:
            print("No histogram data found to delete.")

        return deleted

    def save_normalized_data(self, data: Union[da_core.Array, np.ndarray]) -> None:
        self.dask_data_to_h5({hdf_key_norm_proj: data})

    def save_hist_and_data(
        self,
        hist_output: Union[DaskHistOutput, NpHistOutput],
        data: Union[da_core.Array, np.ndarray],
        base_key: str = hdf_key_ds,
    ) -> None:

        hist, image_range, _, centers, percentile = hist_output
        data_dict = {
            base_key + hdf_key_data: data,
            base_key + hdf_key_bin_frequency: hist[0],
            base_key + hdf_key_bin_edges: hist[1],
            base_key + hdf_key_image_range: image_range,
            base_key + hdf_key_percentile: percentile,
            base_key + hdf_key_bin_centers: centers,
        }
        return self.dask_data_to_h5(data_dict)

    def save_hist(
        self,
        hist_output: Union[DaskHistOutput, NpHistOutput],
    ) -> None:
        hist, image_range, _, centers, percentile = hist_output
        data_dict = {
            hdf_key_norm + hdf_key_bin_frequency: hist[0],
            hdf_key_norm + hdf_key_bin_edges: hist[1],
            hdf_key_norm + hdf_key_image_range: image_range,
            hdf_key_norm + hdf_key_percentile: percentile,
            hdf_key_norm + hdf_key_bin_centers: centers,
        }
        return self.dask_data_to_h5(data_dict)

    @ensure_file_open
    def dask_data_to_h5(
        self, data_dict: dict[str, Union[da_core.Array, np.ndarray]]
    ) -> None:
        """
        Writes lazy dask arrays or numpy arrays to an HDF5 file.

        Parameters:
        ----------
        data_dict: dict
            Dictionary like {"/path/to/data": data}, where `data` can be either a Dask array or a NumPy array.
        """

        print(f"Writing data to {self.filepath} on {self.file}")
        for key, data in data_dict.items():
            # Print debugging information about each dataset being written
            if not isinstance(data, da_core.Array):
                data = da_core.from_array(data)
                data_dict[key] = (
                    data  # Make sure to update the dictionary with the converted array
                )

        # Attempt to write to HDF5 file
        da_core.to_hdf5(self.filepath, data_dict)

    @ensure_file_open
    def check_for_downsampled_data(self) -> bool:
        assert self.file is not None
        return hdf_key_ds in self.file

    # def _return_ds_data(self, pyramid_level=0, px_range=None):

    #     pyramid_level = hdf_key_ds + str(pyramid_level) + "/"
    #     ds_data_key = pyramid_level + hdf_key_data
    #     if px_range is None:
    #         self.data_returned = self.hdf_file[ds_data_key][:]
    #     else:
    #         x = px_range[0]
    #         y = px_range[1]
    #         self.data_returned = self.hdf_file[ds_data_key]
    #         self.data_returned = copy.deepcopy(
    #             self.data_returned[:, y[0] : y[1], x[0] : x[1]]
    #         )

    # def _unload_hdf_normalized_and_ds(self):
    #     self._data = self.hdf_file[hdf_key_norm_proj]
    #     self.data = self._data
    #     pyramid_level = hdf_key_ds + str(0) + "/"
    #     ds_data_key = pyramid_level + hdf_key_data
    #     self.data_ds = self.hdf_file[ds_data_key]

    # def _load_hdf_ds_data_into_memory(self, pyramid_level=0):
    #     pyramid_level = hdf_key_ds + str(pyramid_level) + "/"
    #     ds_data_key = pyramid_level + hdf_key_data
    #     self.data_ds = self.hdf_file[ds_data_key][:]
    #     self.hist = {
    #         key: self.hdf_file[pyramid_level + key][:] for key in hdf_keys_ds_hist
    #     }
    #     for key in hdf_keys_ds_hist_scalar:
    #         self.hist[key] = self.hdf_file[pyramid_level + key][()]
    #     self._data = self.hdf_file[hdf_key_norm_proj]
    #     self.data = self._data
