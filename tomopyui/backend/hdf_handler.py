from tomopyui.backend.io import IOBase
from tomopyui.widgets.hdf_tree import HDF5_Tree

import h5py
import os
import pathlib
import tempfile as tf


class HDF5_Handler:

    # -- Initialization --
    def __init__(self, uploader, hdf_path: pathlib.Path = None, mode="r+"):
        if hdf_path is not None:
            self.hdf_file = h5py.File(hdf_path, mode)
            self.hdf_path = hdf_path
        else:
            self.hdf_file = h5py.File(tf.TemporaryFile(), "w")
            self.hdf_path = None

        self.selected_group_name = None
        self.selected_group = None
        self.selected_dataset_name = None
        self.selected_dataset = None
        self.viewer = uploader.viewer
        self.ds_dropdown = self.viewer.ds_dropdown
        self.tree = HDF5_Tree(self.hdf_file, 0, self)
        self.projections = uploader.projections
        self.uploader = uploader
        self.open_hdf_files = []

    def _init_objs(self):
        self.selected_group_name = None
        self.selected_group = None
        self.selected_dataset_name = None
        self.selected_dataset = None

    def _init_uploader(self, uploader):
        self.uploader = uploader
        self.viewer = uploader.viewer
        self.projections = uploader.projections

    def _init_projections(self, uploader):
        self.projections = uploader.projections

    # -- Opening/closing --
    def new_tree(self, hdf_path):
        self.open_hdf(hdf_path)
        self.tree.__init__(self.hdf_file, 0, self)

    def open_hdf(self, hdf_path: pathlib.Path, mode: str = "r+"):
        self.hdf_path = hdf_path
        try:
            self.hdf_file = h5py.File(hdf_path, mode)
            self.open_hdf_files.append(self.hdf_file)
        except OSError as e:
            print(e)
        except Exception as e:
            print(e)

    def close_hdf(self):
        for file in self.open_hdf_files:
            if file:
                file.close()

    def close_widget(self):
        self.tree.widget.close()

    def close(self):
        self.close_hdf()
        self.close_widget()

    def load_data(self):
        self.find_nearest_process()
        if self.uploader.load_ds_checkbox.value:
            self.load_ds("1")
        else:
            self.load_any()

    def find_nearest_process(self):
        if str(self.selected_group_name).endswith("process"):
            self.nearest_process = self.selected_group
            return
        else:
            for name in self.selected_group_name.parents:
                if str(name).endswith("process"):
                    self.nearest_process = self.hdf_file[str(name)]
                    return
                else:
                    self.nearest_process = None

    def load_ds(self, pyramid_level: str):
        if self.nearest_process is None:
            return
        _ = int(pyramid_level)
        pyramid_level = IOBase.hdf_key_ds + pyramid_level + "/"
        ds_data_key = pyramid_level + IOBase.hdf_key_data
        self.projections.data_ds = self.nearest_process[ds_data_key][:]
        self.projections.hist = {
            key: self.nearest_process[pyramid_level + key][:]
            for key in IOBase.hdf_keys_ds_hist
        }
        for key in IOBase.hdf_keys_ds_hist_scalar:
            self.projections.hist[key] = self.nearest_process[pyramid_level + key][()]
        self.projections._data = self.nearest_process[IOBase.hdf_key_norm_proj]
        self.projections.data = self.projections._data
        self.loaded_ds = True
        self.viewer.plot(self.projections, self)
        self.turn_off_callbacks = True
        self.uploader.viewer.ds_dropdown.value = _
        self.turn_off_callbacks = False
        pyramid_ds_map = {"0": 2, "1": 4, "2": 8}
        self.loaded_ds_factor = pyramid_ds_map[str(_)]

    def load_any(self):
        if self.nearest_process is None:
            return
        strcmp = str(self.selected_group_name)
        grp = str(self.selected_group_name.stem)
        nums = [str(num) for num in range(3)]
        if any([strcmp.endswith(num) for num in nums]):
            self.load_ds(grp)
            return
        elif strcmp.endswith("process"):
            self.load_ds("2")
            return
        elif strcmp.endswith("normalized"):
            self.projections._data = self.nearest_process[IOBase.hdf_key_norm_proj][:]
            self.projections.data = self.projections._data
            pyramid_level = IOBase.hdf_key_ds + str(0) + "/"
            try:
                self.projections.hist = {
                    key: self.nearest_process[IOBase.hdf_key_norm + key][:]
                    for key in IOBase.hdf_keys_ds_hist
                }
            except KeyError:
                # load downsampled histograms if regular histograms don't work
                self.projections.hist = {
                    key: self.nearest_process[pyramid_level + key][:]
                    for key in IOBase.hdf_keys_ds_hist
                }
                for key in IOBase.hdf_keys_ds_hist_scalar:
                    self.projections.hist[key] = self.nearest_process[
                        pyramid_level + key
                    ][()]
            ds_data_key = pyramid_level + IOBase.hdf_key_data
            self.data_ds = self.nearest_process[ds_data_key]
            self.loaded_ds = False
            self.viewer.plot(self.projections, self)
            self.turn_off_callbacks = True
            self.uploader.viewer.ds_dropdown.value = -1
            self.turn_off_callbacks = False
            self.loaded_ds_factor = -1

    def _unload_hdf_normalized_and_ds(self):
        self._data = self.selected_group[self.hdf_key_norm_proj]
        self.data = self._data
        pyramid_level = self.hdf_key_ds + str(0) + "/"
        ds_data_key = pyramid_level + self.hdf_key_data
        self.data_ds = self.hdf_obj[ds_data_key]
