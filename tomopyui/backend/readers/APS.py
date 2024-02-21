from tomopyui.backend.io import RawProjectionsBase, Metadata
import dxchange
import pathlib
import pandas as pd
from tomopyui.backend.schemas import Dxchange


class RawProjectionsHDF5_APS(RawProjectionsBase):
    """
    See RawProjectionsHDF5_ALS832 superclass description.
    """

    def __init__(self):
        super().__init__()
        self.metadata = Metadata_APS_Raw()

    def save_normalized_metadata(self, import_time=None, parent_metadata=None):
        metadata = Metadata_APS_Prenorm()
        metadata.filedir = self.filedir
        if parent_metadata:
            metadata.metadata = parent_metadata.copy()
        if parent_metadata is not None:
            metadata.metadata["parent_metadata"] = parent_metadata.copy()
        if import_time is not None:
            metadata.metadata["import_time"] = import_time
        metadata.set_metadata(self)
        metadata.save_metadata()
        return metadata



class Metadata_APS_Raw(Metadata):
    # Francesco: you will need to edit here.
    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "APS_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.table_label.value = "APS Metadata"

    def set_metadata(self, projections):
        """
        Sets metadata from the APS h5 filetype
        """
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

    def load_metadata_h5(self, h5_filepath: pathlib.Path):
        """
        Loads in metadata from h5 file. You can probably use your dxchange function
        to read all the metadata in at once. Not sure how it works for you.

        The keys in the self.metadata dictionary can be whatever you want, as long as
        your set_attributes_from_metadata function above sets the values correctly.
        """
        # set metadata filepath to the filepath above
        self.filedir = h5_filepath.parent
        self.filepath = h5_filepath

        # Here you will set your metadata. I have left these here from the ALS metadata
        # class for reference. Some things are not inside the metadata (i.e.
        # "energy_units") that I set manually.
        _, meta = dxchange.read_hdf_meta(h5_filepath)
        
        # validate metadata against dxchange schema
        dxchange_meta = Dxchange(**meta)
        
        mapping = {
            "pxY", "/measurement/instrument/detector/dimension_y"
            "pxX", "/measurement/instrument/detector/dimension_x"
            
        }
        _meta = {}
        
        self.metadata["pxY"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_y"
            )[0]
        )
        self.metadata["pxX"] = int(
            dxchange.read_hdf5(
                h5_filepath, "/measurement/instrument/detector/dimension_x"
            )[0]
        )
        self.metadata["pxZ"] = int(
            dxchange.read_hdf5(h5_filepath, "/process/acquisition/rotation/num_angles")[
                0
            ]
        )
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
        
        self.metadata = meta

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


class Metadata_APS_Prenorm(Metadata_APS_Raw):
    """
    Prenormalized metadata class. The table produced by this function may look nearly
    the same for you. For the SSRL version, it looks very different because there is a
    lot of excess information that I store in the SSRL raw metadata file.

    It is important to have this because "import_metadata.json" will be stored in a
    subfolder of the parent, raw data.

    Because the APS prenormalized metadata table looks identical to the raw metadata
    table, I overloaded the create_metadata_box() function to be just an Output widget.

    You can get as fancy as you want with this.

    # Francesco: you will need to edit here.
    """

    def __init__(self):
        super().__init__()
        self.filename: str = "import_metadata.json"
        self.metadata["metadata_type"] = "APS_Normalized"
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

