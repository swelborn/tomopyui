import os
import pathlib
import re

import dask.array as da
import h5py
import numpy as np

from tomopyui.backend.io import Metadata, Projections_Prenormalized

os.environ["cuda_enabled"] = "True"
if os.environ["cuda_enabled"] == "True":
    from tomopyui.tomocupy.prep.alignment import batch_cross_correlation, shift_prj_cp
    from tomopyui.tomocupy.prep.sampling import shrink_and_pad_projections
    from tomopyui.widgets.prep import shift_projections


def get_all_shifts(low_e_prj, high_e_prj):
    sx = np.zeros((low_e_prj.data.shape[0]))
    sy = np.zeros((low_e_prj.data.shape[0]))
    for metadata in high_e_prj.metadatas:
        if metadata.metadata["metadata_type"] == "Align":
            sx += np.array(metadata.metadata["sx"])
            sy += np.array(metadata.metadata["sy"])
    return sx, sy


def shift_projections_init(low_e_prj, high_e_prj_shifted):
    sx, sy = get_all_shifts(low_e_prj, high_e_prj_shifted)
    low_e_prj.data = shrink_and_pad_projections(
        low_e_prj.data,
        low_e_prj.data,
        low_e_prj.energy_float,
        high_e_prj_shifted.energy_float,
        5,
    )
    low_e_prj.data = shift_projections(low_e_prj.data, sx, sy)


high_e = Projections_Prenormalized()
low_e = Projections_Prenormalized()

high_e.filepath = pathlib.Path(
    r"E:\Sam_Welborn\20220620_Welborn\Pristine\all_energies\08375.00eV\20220805-165339-alignment\20220805-1654-SIRT_3D\20220805-1709-alignment\20220805-1714-SIRT_CUDA\normalized_projections.hdf5"
)
high_e.metadatas = Metadata.get_metadata_hierarchy(
    high_e.filedir / "alignment_metadata.json"
)
for metadata in high_e.metadatas:
    metadata.set_attributes_from_metadata(high_e)

filedir = r"E:\Sam_Welborn\20220620_Welborn\Pristine\all_energies"
low_e_filepaths = [pathlib.Path(f) for f in os.scandir(filedir) if f.is_dir()]
hdffile = h5py.File(
    r"E:\Sam_Welborn\20220620_Welborn\Pristine\all_energies\all_energies.hdf5", "w"
)

for filepath in low_e_filepaths:
    low_e.filepath = filepath / "normalized_projections.hdf5"
    low_e._load_hdf_normalized_data_into_memory()
    low_e.metadatas = Metadata.get_metadata_hierarchy(
        low_e.filedir / "import_metadata.json"
    )
    low_e.metadata = low_e.metadatas[0]
    low_e.metadata.set_attributes_from_metadata(low_e)
    shift_projections_init(low_e, high_e)
    group: str = "/energies/" + low_e.energy_str
    grp = hdffile.create_group(group)
    ds = grp.create_dataset("init_shifted", data=low_e.data)

hdffile.close()
