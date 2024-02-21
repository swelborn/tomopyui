from tomopyui.backend.io import Metadata, RawProjectionsBase
import time
import dxchange.exchange
import datetime
import pathlib
import dask.array as da
import pandas as pd
import numpy as np

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

    def import_file_all(self, uploader=None):
        if uploader is None:
            return
        self.import_status_label = uploader.import_status_label
        self.tic = time.perf_counter()
        self.filedir = uploader.filedir
        self.filename = uploader.filename
        self.filepath = self.filedir / self.filename
        self.metadata = uploader.reset_metadata_to()
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
        self.save_data_and_metadata(uploader)

    def import_metadata(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.metadata.load_metadata_h5(filepath)
        self.metadata.set_attributes_from_metadata(self)

    def import_file_projections(self, filepath):
        pass

    def import_file_flats(self, filepath):
        pass

    def import_file_darks(self, filepath):
        pass

    def import_file_angles(self, filepath):
        pass

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

    def save_data_and_metadata(self, uploader):
        """
        Saves current data and metadata in import_savedir.
        """
        self.filedir = self.import_savedir
        self._dask_hist_and_save_data()
        self.saved_as_tiff = False
        _metadata = self.metadata.metadata.copy()
        if uploader.save_tiff_on_import_checkbox.value:
            uploader.import_status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata.metadata["saved_as_tiff"] = True
        self.metadata.filedir = self.filedir
        self.toc = time.perf_counter()
        self.metadata = self.save_normalized_metadata(self.toc - self.tic, _metadata)
        uploader.import_status_label.value = "Checking for downsampled data."
        self._check_downsampled_data(label=uploader.import_status_label)


class Metadata_ALS_832_Raw(Metadata):
    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.table_label.value = "ALS 8.3.2 Metadata"

    def set_metadata(self, projections):

        self.metadata["numslices"] = projections.pxY
        self.metadata["numrays"] = projections.pxX
        self.metadata["num_angles"] = projections.pxZ
        self.metadata["pxsize"] = projections.px_size
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = projections.propagation_dist
        self.metadata["propagation_dist_units"] = "mm"
        self.metadata["angularrange"] = projections.angular_range
        self.metadata["kev"] = projections.energy
        self.metadata["energy_units"] = "keV"
        if projections.angles_deg is not None:
            self.metadata["angles_deg"] = list(projections.angles_deg)
            self.metadata["angles_rad"] = list(projections.angles_rad)

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.propagation_dist = self.metadata["propagation_dist"]
        projections.propagation_dist_units = "mm"
        projections.angular_range = self.metadata["angularrange"]
        projections.energy = self.metadata["kev"]
        projections.units = self.metadata["energy_units"]

    def load_metadata_h5(self, h5_filepath):
        self.filedir = h5_filepath.parent
        self.filepath = h5_filepath
        self.metadata["pxY"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.metadata["numslices"] = self.metadata["pxY"]
        self.metadata["pxX"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.metadata["numrays"] = self.metadata["pxX"]
        self.metadata["pxZ"] = int(
            dxchange.read_hdf5(h5_filepath, "/process/acquisition/rotation/num_angles")[
                0
            ]
        )
        self.metadata["num_angles"] = self.metadata["pxZ"]
        self.metadata["pxsize"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/pixel_size"
            )[0]
            / 10.0
        )  # /10 to convert units from mm to cm
        self.metadata["px_size_units"] = "cm"
        self.metadata["propagation_dist"] = dxchange.read_hdf5(
            h5_filepath,
            "/measurement/instrument/camera_motor_stack/setup/camera_distance",
        )[1]
        self.metadata["energy_float"] = (
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/monochromator/energy"
            )[0]
            / 1000
        )
        self.metadata["kev"] = self.metadata["energy_float"]
        self.metadata["energy_str"] = str(self.metadata["energy_float"])
        self.metadata["energy_units"] = "keV"
        self.metadata["angularrange"] = dxchange.read_hdf5(
            h5_filepath, "/process/acquisition/rotation/range"
        )[0]

    def metadata_to_DataFrame(self):

        # create headers and data for table
        top_headers = []
        middle_headers = []
        data = []
        # Image information
        top_headers.append(["Image Information"])
        middle_headers.append(["X Pixels", "Y Pixels", "Num. θ"])
        data.append(
            [
                self.metadata["numrays"],
                self.metadata["numslices"],
                self.metadata["num_angles"],
            ]
        )

        top_headers.append(["Experiment Settings"])
        middle_headers.append(
            ["Energy (keV)", "Propagation Distance (mm)", "Angular range (deg)"]
        )
        data.append(
            [
                self.metadata["kev"],
                self.metadata["propagation_dist"],
                self.metadata["angularrange"],
            ]
        )

        # create dataframe with the above settings
        df = pd.DataFrame(
            [data[0]],
            columns=pd.MultiIndex.from_product([top_headers[0], middle_headers[0]]),
        )
        for i in range(len(middle_headers)):
            if i == 0:
                continue
            else:
                newdf = pd.DataFrame(
                    [data[i]],
                    columns=pd.MultiIndex.from_product(
                        [top_headers[i], middle_headers[i]]
                    ),
                )
                df = df.join(newdf)

        # set datatable styles
        s = df.style.hide(axis="index")
        s.set_table_styles(
            {
                ("Experiment Settings", "Energy (keV)"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
            },
            overwrite=False,
        )

        s.set_table_styles(
            [
                {"selector": "th.col_heading", "props": "text-align: center;"},
                {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
                {"selector": "td", "props": "text-align: center;" "font-size: 1.2em;"},
                {
                    "selector": "th:not(.index_name)",
                    "props": "background-color: #0F52BA; color: white;",
                },
            ],
            overwrite=False,
        )

        self.dataframe = s


class Metadata_ALS_832_Prenorm(Metadata_ALS_832_Raw):
    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.data_hierarchy_level = self.metadata["data_hierarchy_level"]
        self.table_label.value = ""

    def set_metadata(self, projections):
        super().set_metadata(projections)
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "ALS832_Normalized"
        self.metadata["data_hierarchy_level"] = 1

    def set_attributes_from_metadata(self, projections):
        projections.pxY = self.metadata["numslices"]
        projections.pxX = self.metadata["numrays"]
        projections.pxZ = self.metadata["num_angles"]
        projections.px_size = self.metadata["pxsize"]
        projections.px_size_units = self.metadata["px_size_units"]
        projections.energy = self.metadata["kev"] / 1000
        projections.units = "eV"
        projections.angles_deg = self.metadata["angles_deg"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angle_start = projections.angles_rad[0]
        projections.angle_end = projections.angles_rad[-1]

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):
        """
        Method overloaded because the metadata table is the same as the superclass.
        This avoids a space between tables during display.
        """
        self.metadata_vbox = Output()