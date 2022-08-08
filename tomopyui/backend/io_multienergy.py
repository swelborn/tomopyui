from tomopyui.backend.io import IOBase, Projections_Prenormalized, Metadata, Metadata_MultiEnergy
from tomopyui.tomocupy.prep.alignment import shift_prj_cp, batch_cross_correlation
from tomopyui.tomocupy.prep.sampling import shrink_and_pad_projections
from tomopyui.widgets.prep import shift_projections

import numpy as np

import pathlib
import h5py
import os

class MultiEnergyProjections(IOBase):

    # hdf keys
    hdf_key_theta = "/exchange/theta"
    hdf_key_energies = "/energies"
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

    def __init__(self, hdf_path: pathlib.Path):
        self.energies: list[float] = []
        self.hdf_path = hdf_path
        self.this_energy_projections = Projections_Prenormalized()
        self.metadata = Metadata_MultiEnergy()

    def compile_energies(
        self, folders: list, alignment_meta: pathlib.Path, write_location: pathlib.Path
    ):
        """
        Compiles energies into one HDF5 file and aligns all data to a pre-aligned upper
        dataset. This should account for a significant amount of the "wobble" in the
        data. This also rescales to the upper-most energy

        Parameters
        ----------
        folders: list of pathlib.Path
            Paths to folders that contain imported data from XANES tomography.

        alignment_meta: pathlib.Path
            Path to alignment metadata for the pre-aligned dataset. This could be a
            "standard," or the highest energy dataset for the sample. Must have
            alignment_metadata.json in folder.

        write_location: pathlib.Path
            Location where compiled energy hdf5 file will be written.

        """

        # Setting the shift values
        ref = Projections_Prenormalized()
        ref.filepath = alignment_meta
        ref.metadatas = Metadata.get_metadata_hierarchy(
            ref.filedir / "alignment_metadata.json"
        )
        for metadata in ref.metadatas:
            metadata.set_attributes_from_metadata(ref)
        ref._unload_hdf_normalized_and_ds()
        ref_shape = ref.data.shape

        sx = np.zeros((ref_shape[0]))
        sy = np.zeros((ref_shape[0]))
        for metadata in ref.metadatas:
            if metadata.metadata["metadata_type"] == "Align":
                sx += np.array(metadata.metadata["sx"])
                sy += np.array(metadata.metadata["sy"])

        # Shifting all of the lower energies and writing
        hdf_file = h5py.File(write_location, "a")

        self.energies = []
        self.energies_str = []
        for folder in folders:
            moving = Projections_Prenormalized()
            moving.filepath = folder / "normalized_projections.hdf5"
            moving._load_hdf_normalized_data_into_memory()
            moving.metadatas = Metadata.get_metadata_hierarchy(
                moving.filedir / "import_metadata.json"
            )
            moving.metadata = moving.metadatas[0]
            moving.metadata.set_attributes_from_metadata(moving)
            moving.data = shrink_and_pad_projections(
                moving.data, ref.data, moving.energy_float, ref.energy_float, 5
            )
            moving.data = shift_projections(moving.data, sx, sy)
            group: str = self.hdf_key_energies + "/" + moving.energy_str
            if group + self.hdf_key_norm not in hdf_file:
                grp = hdf_file.create_group(group + self.hdf_key_norm)
                ds = grp.create_dataset(self.hdf_key_data, data=moving.data)
            else:
                grp = hdf_file[group + self.hdf_key_norm]
                ds = grp[self.hdf_key_data]
                ds[...] = moving.data
            ds.attrs["energy"] = moving.energy_float
            ds.attrs["sx"] = sx
            ds.attrs["sy"] = sy
            for key in self.hdf_keys_ds_hist:
                ds = grp.create_dataset(key, data=moving.hist[key])
            self.energies.append(moving.energy_float)
            self.energies_str.append(moving.energy_str)

            # Padding, shifting, saving downsampled + histograms
            for i in range(3):
                sx = [x / 2 for x in sx]
                sy = [y / 2 for y in sy]
                moving._load_hdf_ds_data_into_memory(pyramid_level=i)
                ref._load_hdf_ds_data_into_memory(pyramid_level=i)
                moving.data_ds = shrink_and_pad_projections(
                    moving.data_ds,
                    ref.data_ds,
                    moving.energy_float,
                    ref.energy_float,
                    5,
                )
                moving.data_ds = shift_projections(moving.data_ds, sx, sy)
                if group + self.hdf_key_ds + "/" + str(i) + "/" not in hdf_file:
                    grp = hdf_file.create_group(group + self.hdf_key_ds + "/" + str(i) + "/")
                    ds = grp.create_dataset(self.hdf_key_data, data=moving.data_ds)
                else:
                    grp = hdf_file[group + self.hdf_key_ds + "/" + str(i) + "/"]
                    ds = grp[self.hdf_key_data]
                    ds[...] = moving.data
                ds.attrs["energy"] = moving.energy_float
                ds.attrs["sx"] = sx
                ds.attrs["sy"] = sy
                ds.attrs["angles_deg"] = moving.metadata.metadata["angles_deg"]
                for key in self.hdf_keys_ds_hist:
                    ds = grp.create_dataset(key, data=moving.hist[key])
                for key in self.hdf_keys_ds_hist_scalar:
                    ds = grp.create_dataset(key, data=moving.hist[key])

            moving._close_hdf_file()

        ref._close_hdf_file()
        hdf_file[self.hdf_key_energies].attrs["energies_float"] = self.energies
        hdf_file[self.hdf_key_energies].attrs["energies_str"] = self.energies_str
        hdf_file.close()

    def get_folders(self, filedir: pathlib.Path):
        """
        Grabs all folders in a file directory.

        Parameters
        ----------
        filedir: pathlib.Path
            File directory to grab from

        Returns
        -------
        folders: list[pathlib.Path]
            Returns a list of all folders as pathlib.Paths.

        """
        folders = [pathlib.Path(f) for f in os.scandir(filedir) if f.is_dir()]
        return folders

    def import_file_projections(self, filepath: pathlib.Path):
        """
        Imports hdf5 file with multiple energy projections
        Parameters
        ----------
        filepath: pathlib.Path
            Path to hdf5 file creating using compile_energies.
        

        """
        # with h5py.File(filepath) as f:
        #     self.init_energy = 
        #     self.hdf_key_norm_data = 
        #     self.
        pass