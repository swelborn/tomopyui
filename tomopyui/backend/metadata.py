class Metadata(ABC):
    """
    Base class for all metadatas.
    """

    def __init__(self):
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }
        self.table_label = Label(style=self.header_font_style)
        self.metadata = {}
        self.filedir = None
        self.filename = None
        self.filepath = None

    def save_metadata(self):
        with open(self.filedir / self.filename, "w+") as f:
            a = safe_serialize(self.metadata, f)

    def load_metadata(self):
        with open(self.filepath) as f:
            self.metadata = json.load(f)

        return self.metadata

    def set_parent_metadata(self, parent_metadata):
        self.metadata["parent_metadata"] = parent_metadata
        self.metadata["data_hierarchy_level"] = (
            parent_metadata["data_hierarchy_level"] + 1
        )

    def create_metadata_box(self):
        """
        Creates the box to be displayed on the frontend when importing data. Has both
        a label and the metadata dataframe (stored in table_output).

        """
        self.metadata_to_DataFrame()
        self.table_output = Output()
        if self.dataframe is not None:
            with self.table_output:
                display(self.dataframe)
        self.metadata_vbox = VBox(
            [self.table_label, self.table_output], layout=Layout(align_items="center")
        )

    @staticmethod
    def parse_metadata_type(filepath: pathlib.Path = None, metadata=None):
        """
        Determines the type of metadata by looking at the "metadata_type" key in the
        loaded dictionary.

        Parameters
        ----------
        filepath: pathlib.Path
            Filepath for the metadata. If this is not specified, metadata should be
            specified
        metadata: dict
            A metadata dictionary with the "metadata_type" key. If this is not
            specified, a filepath should be specified.

        Returns
        -------
        A metadata instance with the metadata.

        """
        if filepath is not None:
            with open(filepath) as f:
                metadata = json.load(f)

        if "metadata_type" not in metadata:
            metadata["metadata_type"] = "SSRL62C_Normalized"

        # General Data
        if metadata["metadata_type"] == "General_Normalized":
            metadata_instance = Metadata_General_Prenorm()

        # SSRL Beamlines
        if metadata["metadata_type"] == "SSRL62C_Normalized":
            metadata_instance = Metadata_SSRL62C_Prenorm()
        if metadata["metadata_type"] == "SSRL62C_Raw":
            metadata_instance = Metadata_SSRL62C_Raw()
        if metadata["metadata_type"] == "SSRL62B_Normalized":
            metadata_instance = Metadata_SSRL62B_Prenorm()
        if metadata["metadata_type"] == "SSRL62B_Raw":
            metadata_instance = Metadata_SSRL62B_Raw()

        # ALS Beamlines
        if metadata["metadata_type"] == "ALS832_Normalized":
            metadata_instance = Metadata_ALS_832_Prenorm()
        if metadata["metadata_type"] == "ALS832_Raw":
            metadata_instance = Metadata_ALS_832_Raw()

        # Metadata through rest of processing pipeline
        if metadata["metadata_type"] == "Prep":
            metadata_instance = Metadata_Prep()
        if metadata["metadata_type"] == "Align":
            metadata_instance = Metadata_Align()
        if metadata["metadata_type"] == "Recon":
            metadata_instance = Metadata_Recon()

        if metadata["metadata_type"] == "2E":
            metadata_instance = Metadata_TwoE()

        if filepath is not None:
            metadata_instance.filedir = filepath.parent
            metadata_instance.filename = filepath.name
            metadata_instance.filepath = filepath

        return metadata_instance

    @staticmethod
    def get_metadata_hierarchy(filepath):
        """
        Reads in a metadata file from filepath and determines its hierarchy. Generates
        a list of `Metadata` instances, found by Metadata.parse_metadata_type.

        Parameters
        ----------
        filepath: pathlike
            Metadata file path.

        Returns
        -------
        metadata_insts: list(`Metadata`)
            List of metadata instances associated with the metadata file.
        """
        with open(filepath) as f:
            metadata = json.load(f)
        num_levels = metadata["data_hierarchy_level"]
        metadata_insts = []
        for i in range(num_levels + 1):
            metadata_insts.append(
                Metadata.parse_metadata_type(metadata=metadata.copy())
            )
            metadata_insts[i].metadata = metadata.copy()
            if "parent_metadata" in metadata:
                metadata = metadata["parent_metadata"].copy()
        return metadata_insts

    @abstractmethod
    def set_metadata(self, projections):
        """
        Sets metadata from projections attributes.
        """
        ...

    @abstractmethod
    def metadata_to_DataFrame(self):
        """
        This will take the metadata that you have and turn it into a table for display
        on the frontend. It is a little complicated, but I don't know pandas very well.
        You will have "top_headers" which are the headers at the top of the table like
        "Image Information". The subheaders are called "middle_headers": things like
        the X Pixels, Y Pixels, and the number of angles. Then below each of the middle
        headers, you have the data. The dimensions of each should match up properly

        This creates a dataframe and then s.set_table_styles() styles it. This styling
        function is based on CSS, which I know very little about. You can make the
        table as fancy as you want, but for now I just have a blue background header
        and white lines dividing the major table sections.
        """
        ...


class Metadata_General_Prenorm(Metadata):
    """
    General prenormalized metadata. This will be created if you are importing a tiff
    or tiff stack, or npy file that was not previously imported using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "General_Normalized"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.imported = False
        self.table_label.value = "User Metadata"

    def set_metadata(self, projections):
        pass

    def metadata_to_DataFrame(self):
        # create headers and data for table
        self.metadata["energy_str"] = f"{self.metadata['energy_float']:0.2f}"
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        ang_res = self.metadata["angular_resolution"]
        self.metadata["num_angles"] = int((end_angle - start_angle) / ang_res)

        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                "Angular Resolution (°)": f"{ang_res:0.2f}",
            },
            {
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Binning": self.metadata["binning"],
                "Num. θ (est)": self.metadata["num_angles"],
            },
        ]
        if "pxX" in self.metadata:
            self.metadata_list_for_table[1]["X Pixels"] = self.metadata["pxX"]
            self.metadata_list_for_table[1]["Y Pixels"] = self.metadata["pxY"]
            self.metadata_list_for_table[1]["Num. θ"] = self.metadata["pxZ"]
            self.make_angles_from_metadata()

        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])

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
                # ("Acquisition Information", middle_headers[0][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
                ("Image Information", middle_headers[1][0]): [
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

    def set_attributes_from_metadata_before_import(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy = projections.energy_float
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]

    def set_metadata_from_attributes_after_import(self, projections):
        self.metadata["normalized_projections_size_gb"] = projections.size_gb
        self.metadata["normalized_projections_directory"] = str(
            projections.import_savedir
        )
        if "filedir_ds" in projections.__dict__:
            self.metadata["downsampled_projections_directory"] = str(
                projections.filedir_ds
            )
        self.metadata["saved_as_tiff"] = projections.saved_as_tiff
        self.metadata["num_angles"] = projections.data.shape[0]
        self.metadata["pxX"] = projections.data.shape[2]
        self.metadata["pxY"] = projections.data.shape[1]
        self.metadata["pxZ"] = projections.data.shape[0]

    def set_attributes_from_metadata(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy = projections.energy_float
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        projections.size_gb = self.metadata["normalized_projections_size_gb"]
        projections.import_savedir = pathlib.Path(
            self.metadata["normalized_projections_directory"]
        )
        if "downsampled_projections_directory" in self.metadata:
            projections.filedir_ds = pathlib.Path(
                self.metadata["downsampled_projections_directory"]
            )
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]
        if "angles_rad" in self.metadata:
            projections.angles_rad = self.metadata["angles_rad"]
            projections.angles_deg = self.metadata["angles_deg"]

    def make_angles_from_metadata(self):
        self.metadata["angles_rad"] = angle_maker(
            self.metadata["pxZ"],
            ang1=self.metadata["start_angle"],
            ang2=self.metadata["end_angle"],
        )
        self.metadata["angles_rad"] = list(self.metadata["angles_rad"])
        self.metadata["angles_deg"] = [
            x * 180 / np.pi for x in self.metadata["angles_rad"]
        ]


class Metadata_SSRL62C_Raw(Metadata):
    """
    Raw metadata from SSRL 6-2C. Will be created if you import a folder filled with
    raw XRMs.
    """

    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62C_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2C Raw Metadata"

    def set_attributes_from_metadata(self, projections):
        pass

    def set_metadata(self, projections):
        self.metadata["scan_info"] = copy.deepcopy(projections.scan_info)
        self.metadata["scan_info"]["FILES"] = [
            str(file) for file in projections.scan_info["FILES"]
        ]
        self.metadata["scan_info_path"] = str(projections.scan_info_path)
        self.metadata["run_script_path"] = str(projections.run_script_path)
        self.metadata["flats_filenames"] = [
            str(file) for file in projections.flats_filenames
        ]
        self.metadata["projections_filenames"] = [
            str(file) for file in projections.data_filenames
        ]
        self.metadata["scan_type"] = projections.scan_type
        self.metadata["scan_order"] = projections.scan_order
        self.metadata["pxX"] = projections.pxX
        self.metadata["pxY"] = projections.pxY
        self.metadata["pxZ"] = projections.pxZ
        self.metadata["num_angles"] = projections.pxZ
        self.metadata["angles_rad"] = projections.angles_rad
        self.metadata["angles_deg"] = projections.angles_deg
        self.metadata["start_angle"] = float(projections.angles_deg[0])
        self.metadata["end_angle"] = float(projections.angles_deg[-1])
        self.metadata["binning"] = projections.binning
        if isinstance(projections.scan_info["PROJECTION_METADATA"], list):
            self.metadata["projections_exposure_time"] = projections.scan_info[
                "PROJECTION_METADATA"
            ][0]["exposure_time"]
        else:
            self.metadata["projections_exposure_time"] = projections.scan_info[
                "PROJECTION_METADATA"
            ]["exposure_time"]
        if isinstance(projections.scan_info["FLAT_METADATA"], list):
            self.metadata["references_exposure_time"] = projections.scan_info[
                "FLAT_METADATA"
            ][0]["exposure_time"]
        else:
            self.metadata["references_exposure_time"] = projections.scan_info[
                "FLAT_METADATA"
            ]["exposure_time"]

        self.metadata["all_raw_energies_float"] = projections.energies_list_float
        self.metadata["all_raw_energies_str"] = projections.energies_list_str
        self.metadata["all_raw_pixel_sizes"] = projections.raw_pixel_sizes
        self.metadata[
            "pixel_size_from_scan_info"
        ] = projections.pixel_size_from_metadata
        self.metadata["energy_units"] = "eV"
        self.metadata["pixel_units"] = "nm"
        self.metadata["raw_projections_dtype"] = str(projections.raw_data_type)
        self.metadata["raw_projections_directory"] = str(
            projections.data_filenames[0].parent
        )
        self.metadata["data_hierarchy_level"] = projections.data_hierarchy_level

    def metadata_to_DataFrame(self):

        # change metadata keys to be better looking
        if self.metadata["scan_info"]["VERSION"] == 1:
            keys = {
                "ENERGY": "Energy",
                "TOMO": "Tomo",
                "MOSAIC": "Mosaic",
                "MULTIEXPOSURE": "MultiExposure",
                "NREPEATSCAN": "Repeat Scan",
                "WAITNSECS": "Wait (s)",
                "NEXPOSURES": "Num. Exposures",
                "AVERAGEONTHEFLY": "Average On the Fly",
                "REFNEXPOSURES": "Num. Ref Exposures",
                "REFEVERYEXPOSURES": "Ref/Num Exposures",
                "REFABBA": "Order",
                "MOSAICUP": "Up",
                "MOSAICDOWN": "Down",
                "MOSAICLEFT": "Left",
                "MOSAICRIGHT": "Right",
                "MOSAICOVERLAP": "Overlap (%)",
                "MOSAICCENTRALTILE": "Central Tile",
            }
        if self.metadata["scan_info"]["VERSION"] == 2:
            keys = {
                "ENERGY": "Energy",
                "TOMO": "Tomo",
                "MOSAIC": "Mosaic",
                "MULTIEXPOSURE": "MultiExposure",
                "NREPEATSCAN": "Repeat Scan",
                "WAITNSECS": "Wait (s)",
                "NEXPOSURES": "Num. Exposures",
                "AVERAGEONTHEFLY": "Average On the Fly",
                "IMAGESPERPROJECTION": "Images/Projection",
                "REFNEXPOSURES": "Num. Ref Exposures",
                "REFEVERYEXPOSURES": "Ref/Num Exposures",
                "REFABBA": "Order",
                "REFDESPECKLEAVERAGE": "Ref Despeckle Avg",
                "APPLYREF": "Ref Applied",
                "MOSAICUP": "Up",
                "MOSAICDOWN": "Down",
                "MOSAICLEFT": "Left",
                "MOSAICRIGHT": "Right",
                "MOSAICOVERLAP": "Overlap (%)",
                "MOSAICCENTRALTILE": "Central Tile",
            }
        m = {keys[key]: self.metadata["scan_info"][key] for key in keys}

        if m["Order"] == 0:
            m["Order"] = "ABAB"
        else:
            m["Order"] = "ABBA"

        # create headers and data for table
        middle_headers = []
        middle_headers.append(["Energy", "Tomo", "Mosaic", "MultiExposure"])
        if self.metadata["scan_info"]["VERSION"] == 1:
            middle_headers.append(
                [
                    "Repeat Scan",
                    "Wait (s)",
                    "Num. Exposures",
                ]
            )
            middle_headers.append(["Num. Ref Exposures", "Ref/Num Exposures", "Order"])
        if self.metadata["scan_info"]["VERSION"] == 2:
            middle_headers.append(
                [
                    "Repeat Scan",
                    "Wait (s)",
                    "Num. Exposures",
                    "Images/Projection",
                ]
            )
            middle_headers.append(
                [
                    "Num. Ref Exposures",
                    "Ref/Num Exposures",
                    "Order",
                    "Ref Despeckle Avg",
                ]
            )
        middle_headers.append(["Up", "Down", "Left", "Right"])
        top_headers = []
        top_headers.append(["Layers"])
        top_headers.append(["Image Information"])
        top_headers.append(["Acquisition Information"])
        top_headers.append(["Reference Information"])
        top_headers.append(["Mosaic Information"])
        data = [
            [m[key] for key in middle_headers[i]] for i in range(len(middle_headers))
        ]
        middle_headers.insert(1, ["X Pixels", "Y Pixels", "Num. θ"])
        data.insert(
            1,
            [
                self.metadata["pxX"],
                self.metadata["pxY"],
                len(self.metadata["angles_rad"]),
            ],
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
                ("Acquisition Information", "Repeat Scan"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Image Information", "X Pixels"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Mosaic Information", "Up"): [
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


class Metadata_SSRL62C_Prenorm(Metadata_SSRL62C_Raw):
    """
    Metadata class for data from SSRL 6-2C that was normalized using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "SSRL62C_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.table_label.value = "SSRL 6-2C TomoPyUI-Imported Metadata"

    def set_metadata(self, projections):
        super().set_metadata(projections)
        metadata_to_remove = [
            "scan_info_path",
            "run_script_path",
            "scan_info",
            "scan_type",
            "scan_order",
            "all_raw_energies_float",
            "all_raw_energies_str",
            "all_raw_pixel_sizes",
            "pixel_size_from_scan_info",
            "raw_projections_dtype",
        ]
        # removing unneeded things from parent raw
        [
            self.metadata.pop(name)
            for name in metadata_to_remove
            if name in self.metadata
        ]
        self.metadata["flats_ind"] = projections.flats_ind
        self.metadata["user_overwrite_energy"] = projections.user_overwrite_energy
        self.metadata["energy_str"] = projections.energy_str
        self.metadata["energy_float"] = projections.energy_float
        self.metadata["pixel_size"] = projections.px_size
        self.metadata["normalized_projections_dtype"] = str(np.dtype(np.float32))
        self.metadata["normalized_projections_size_gb"] = projections.size_gb
        self.metadata["normalized_projections_directory"] = str(
            projections.import_savedir
        )
        self.metadata[
            "normalized_projections_filename"
        ] = projections.normalized_projections_hdf_key
        self.metadata["normalization_function"] = "dask"
        self.metadata["saved_as_tiff"] = projections.saved_as_tiff

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        if isinstance(self.metadata["projections_exposure_time"], list):
            exp_time_proj = f"{self.metadata['projections_exposure_time'][0]:0.2f}"
        else:
            exp_time_proj = f"{self.metadata['projections_exposure_time']:0.2f}"
        if isinstance(self.metadata["references_exposure_time"], list):
            exp_time_ref = f"{self.metadata['references_exposure_time'][0]:0.2f}"
        else:
            exp_time_ref = f"{self.metadata['references_exposure_time']:0.2f}"
        if self.metadata["user_overwrite_energy"]:
            user_overwrite = "Yes"
        else:
            user_overwrite = "No"
        if self.metadata["saved_as_tiff"]:
            save_as_tiff = "Yes"
        else:
            save_as_tiff = "No"
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                # "Scan Type": self.metadata["scan_type"],
                "Ref. Exp. Time": exp_time_ref,
                "Proj. Exp. Time": exp_time_proj,
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. θ": self.metadata["num_angles"],
                "Binning": self.metadata["binning"],
            },
            {
                "Energy Overwritten": user_overwrite,
                ".tif Saved": save_as_tiff,
            },
        ]
        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])
        top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Other Information", middle_headers[2][0]): [
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

    def set_attributes_from_metadata(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.user_overwrite_energy = self.metadata["user_overwrite_energy"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        # projections.import_savedir = pathlib.Path(
        #     self.metadata["normalized_projections_directory"]
        # )
        if "downsampled_projections_directory" in self.metadata:
            projections.filedir_ds = pathlib.Path(
                self.metadata["downsampled_projections_directory"]
            )
        if "flats_ind" in self.metadata:
            projections.flats_ind = self.metadata["flats_ind"]
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]


class Metadata_SSRL62B_Raw_Projections(Metadata):
    """
    Raw projections metadata from SSRL 6-2B.
    """

    summary_key = "Summary"
    coords_default_key = r"Coords-Default/"
    metadata_default_key = r"Metadata-Default/"

    def __init__(self):
        super().__init__()
        self.loaded_metadata = False  # did we load metadata yet? no
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw_Projections"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw Projections Metadata"

    def parse_raw_metadata(self):
        self.load_metadata()
        self.summary = self.imported_metadata["Summary"].copy()
        self.metadata["acquisition_name"] = self.summary["Prefix"]
        self.metadata["angular_resolution"] = self.summary["z-step_um"] / 1000
        self.metadata["pxZ"] = self.summary["Slices"]
        self.metadata["num_angles"] = self.metadata["pxZ"]
        self.metadata["pixel_type"] = self.summary["PixelType"]
        self.meta_keys = [
            key for key in self.imported_metadata.keys() if "Metadata-Default" in key
        ]
        self.metadata["angles_deg"] = [
            self.imported_metadata[key]["ZPositionUm"] / 1000 for key in self.meta_keys
        ]
        self.metadata["angles_rad"] = [
            x * np.pi / 180 for x in self.metadata["angles_deg"]
        ]
        self.metadata["start_angle"] = self.metadata["angles_deg"][0]
        self.metadata["end_angle"] = self.metadata["angles_deg"][-1]
        self.metadata["exposure_times_ms"] = [
            self.imported_metadata[key]["Exposure-ms"] for key in self.meta_keys
        ]
        self.metadata["average_exposure_time"] = np.mean(
            self.metadata["exposure_times_ms"]
        )
        self.metadata["elapsed_times_ms"] = [
            self.imported_metadata[key]["ElapsedTime-ms"] for key in self.meta_keys
        ]
        self.metadata["received_times"] = [
            self.imported_metadata[key]["ReceivedTime"] for key in self.meta_keys
        ]
        self.metadata["filenames"] = [
            key.replace(r"Metadata-Default/", "") for key in self.meta_keys
        ]
        self.metadata["widths"] = [
            self.imported_metadata[key]["Width"] for key in self.meta_keys
        ]
        self.metadata["heights"] = [
            self.imported_metadata[key]["Height"] for key in self.meta_keys
        ]
        self.metadata["binnings"] = [
            self.imported_metadata[key]["Binning"] for key in self.meta_keys
        ]
        self.metadata["pxX"] = self.metadata["heights"][0]
        self.metadata["pxY"] = self.metadata["widths"][0]
        self.loaded_metadata = True

    def set_extra_metadata(self, Uploader):
        self.metadata["energy_float"] = Uploader.energy_textbox.value
        self.metadata["energy_str"] = f"{self.metadata['energy_float']:0.2f}"
        self.metadata["energy_units"] = Uploader.energy_units_dropdown.value
        self.metadata["pixel_size"] = Uploader.px_size_textbox.value
        self.metadata["pixel_units"] = Uploader.px_units_dropdown.value

    def load_metadata(self):
        with open(self.filepath) as f:
            self.imported_metadata = json.load(f)
        return self.imported_metadata

    def set_attributes_from_metadata(self, projections):
        projections.binning = self.metadata["binnings"][0]
        projections.num_angles = self.metadata["num_angles"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.start_angle = self.metadata["start_angle"]
        projections.pxZ = self.metadata["pxZ"]
        projections.pxY = self.metadata["pxY"]
        projections.pxX = self.metadata["pxX"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_units = self.metadata["energy_units"]
        projections.pixel_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        projections.projections_exposure_time = self.metadata["average_exposure_time"]
        projections.acquisition_name = self.metadata["acquisition_name"]

    def set_metadata(self, projections):
        pass

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                "Exp. Time (ms)": f"{self.metadata['average_exposure_time']:0.2f}",
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. θ": self.metadata["num_angles"],
                "Binning": self.metadata["binnings"][0],
            },
        ]
        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])
        # top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                # ("Other Information", middle_headers[2][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
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


class Metadata_SSRL62B_Raw_References(Metadata_SSRL62B_Raw_Projections):
    """
    Raw reference metadata from SSRL 6-2B.
    """

    def __init__(self):
        super().__init__()
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw_References"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw References Metadata"

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                # "Start θ (°)": f"{start_angle:0.1f}",
                # "End θ (°)": f"{end_angle:0.1f}",
                "Exp. Time (ms)": f"{self.metadata['average_exposure_time']:0.2f}",
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. Refs": len(self.metadata["widths"]),
                "Binning": self.metadata["binnings"][0],
            },
        ]
        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])
        # top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                # ("Other Information", middle_headers[2][0]): [
                #     {"selector": "td", "props": "border-left: 1px solid white"},
                #     {"selector": "th", "props": "border-left: 1px solid white"},
                # ],
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


class Metadata_SSRL62B_Raw(Metadata_SSRL62B_Raw_Projections):
    """
    Raw reference metadata from SSRL 6-2B.
    """

    def __init__(self, metadata_projections, metadata_references):
        super().__init__()
        self.metadata_projections = metadata_projections
        self.metadata_references = metadata_references
        self.metadata["projections_metadata"] = self.metadata_projections.metadata
        self.metadata["references_metadata"] = self.metadata_references.metadata
        self.filename = "raw_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Raw"
        self.metadata["data_hierarchy_level"] = 0
        self.data_hierarchy_level = 0
        self.table_label.value = "SSRL 6-2B Raw Metadata"

    def metadata_to_DataFrame(self):
        # create headers and data for table
        self.metadata_projections.create_metadata_box()
        self.metadata_references.create_metadata_box()

    def create_metadata_hbox(self):
        """
        Creates the box to be displayed on the frontend when importing data. Has both
        a label and the metadata dataframe (stored in table_output).

        """
        self.metadata_to_DataFrame()
        self.table_output = Output()
        if (
            self.metadata_projections.dataframe is not None
            and self.metadata_references.dataframe is not None
        ):
            self.metadata_hbox = HBox(
                [
                    self.metadata_projections.metadata_vbox,
                    self.metadata_references.metadata_vbox,
                ],
                layout=Layout(justify_content="center"),
            )


class Metadata_SSRL62B_Prenorm(Metadata_SSRL62B_Raw_Projections):
    """
    Metadata class for data from SSRL 6-2C that was normalized using TomoPyUI.
    """

    def __init__(self):
        super().__init__()
        self.filename = "import_metadata.json"
        self.metadata["metadata_type"] = "SSRL62B_Normalized"
        self.metadata["data_hierarchy_level"] = 1
        self.table_label.value = "SSRL 6-2B TomoPyUI-Imported Metadata"

    def set_metadata(self, projections):
        self.metadata["num_angles"] = projections.num_angles
        self.metadata["angles_deg"] = projections.angles_deg
        self.metadata["angles_rad"] = projections.angles_rad
        self.metadata["start_angle"] = projections.start_angle
        self.metadata["end_angle"] = projections.end_angle
        self.metadata["start_angle"] = projections.start_angle
        self.metadata["pxZ"] = projections.pxZ
        self.metadata["pxY"] = projections.pxY
        self.metadata["pxX"] = projections.pxX
        self.metadata["energy_float"] = projections.energy_float
        self.metadata["energy_str"] = projections.energy_str
        self.metadata["energy_units"] = projections.energy_units
        self.metadata["pixel_size"] = projections.pixel_size
        self.metadata["pixel_units"] = projections.pixel_units
        self.metadata["binning"] = projections.binning
        self.metadata["average_exposure_time"] = projections.projections_exposure_time
        self.metadata["acquisition_name"] = projections.acquisition_name
        self.metadata["saved_as_tiff"] = projections.saved_as_tiff

    def metadata_to_DataFrame(self):
        # create headers and data for table
        px_size = self.metadata["pixel_size"]
        px_units = self.metadata["pixel_units"]
        en_units = self.metadata["energy_units"]
        start_angle = self.metadata["start_angle"]
        end_angle = self.metadata["end_angle"]
        exp_time_proj = f"{self.metadata['average_exposure_time']:0.2f}"
        if self.metadata["saved_as_tiff"]:
            save_as_tiff = "Yes"
        else:
            save_as_tiff = "No"
        self.metadata_list_for_table = [
            {
                f"Energy ({en_units})": self.metadata["energy_str"],
                f"Pixel Size ({px_units})": f"{px_size:0.2f}",
                "Start θ (°)": f"{start_angle:0.1f}",
                "End θ (°)": f"{end_angle:0.1f}",
                # "Scan Type": self.metadata["scan_type"],
                "Proj. Exp. Time": exp_time_proj,
            },
            {
                "X Pixels": self.metadata["pxX"],
                "Y Pixels": self.metadata["pxY"],
                "Num. θ": self.metadata["num_angles"],
                "Binning": self.metadata["binning"],
            },
            {
                ".tif Saved": save_as_tiff,
            },
        ]
        middle_headers = [[]]
        data = [[]]
        for i in range(len(self.metadata_list_for_table)):
            middle_headers.append([key for key in self.metadata_list_for_table[i]])
            data.append(
                [
                    self.metadata_list_for_table[i][key]
                    for key in self.metadata_list_for_table[i]
                ]
            )
        data.pop(0)
        middle_headers.pop(0)
        top_headers = [["Acquisition Information"]]
        top_headers.append(["Image Information"])
        top_headers.append(["Other Information"])

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
                ("Image Information", middle_headers[1][0]): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ],
                ("Other Information", middle_headers[2][0]): [
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

    def set_attributes_from_metadata(self, projections):
        projections.pxX = self.metadata["pxX"]
        projections.pxY = self.metadata["pxY"]
        projections.pxZ = self.metadata["pxZ"]
        projections.angles_rad = self.metadata["angles_rad"]
        projections.angles_deg = self.metadata["angles_deg"]
        projections.start_angle = self.metadata["start_angle"]
        projections.end_angle = self.metadata["end_angle"]
        projections.binning = self.metadata["binning"]
        projections.energy_str = self.metadata["energy_str"]
        projections.energy_float = self.metadata["energy_float"]
        projections.energy_units = self.metadata["energy_units"]
        projections.px_size = self.metadata["pixel_size"]
        projections.pixel_units = self.metadata["pixel_units"]
        projections.saved_as_tiff = self.metadata["saved_as_tiff"]
        projections.num_angles = self.metadata["num_angles"]
        projections.acquisition_name = self.metadata["acquisition_name"]
        projections.exposure_time = self.metadata["average_exposure_time"]


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

    def load_metadata_h5(self, h5_filepath):
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
        self.filename = "import_metadata.json"
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


class Metadata_Prep(Metadata):
    def __init__(self):
        super().__init__()
        self.table_label.value = "Preprocessing Methods"
        self.prep_list_label_style = {
            "font_size": "16px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }

    def set_metadata(self, Prep):
        self.metadata["metadata_type"] = "Prep"
        self.filename = "prep_metadata.json"
        self.parent_metadata = Prep.Import.projections.metadata
        self.metadata["parent_metadata"] = self.parent_metadata.metadata
        if "data_hierarchy_level" in self.parent_metadata.metadata:
            self.metadata["data_hierarchy_level"] = (
                self.parent_metadata.metadata["data_hierarchy_level"] + 1
            )
        else:
            self.metadata["data_hierarchy_level"] = 2
        self.metadata["prep_list"] = [
            (x[1].method_name, x[1].opts) for x in Prep.prep_list
        ]
        self.table_label.value = "Preprocessing Metadata"

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):
        display_str = [x[0] + " → " for x in self.metadata["prep_list"][:-1]]
        display_str = "".join(display_str + [self.metadata["prep_list"][-1][0]])

        self.prep_list_label = Label(display_str, style=self.prep_list_label_style)
        self.metadata_vbox = VBox(
            [self.table_label, self.prep_list_label],
            layout=Layout(align_items="center"),
        )

    def set_attributes_from_metadata(self, projections):
        pass


class Metadata_TwoE(Metadata):
    def __init__(self):
        super().__init__()
        self.table_label.value = "Preprocessing Methods"
        self.prep_list_label_style = {
            "font_size": "16px",
            "font_weight": "bold",
            "font_variant": "small-caps",
            # "text_color": "#0F52BA",
        }

    def set_metadata(self, TwoEnergyTool):
        self.metadata["metadata_type"] = "2E"
        self.filename = "2E_metadata.json"
        self.parent_metadata = TwoEnergyTool.low_e_viewer.projections.metadatas[0].metadata.copy()
        self.high_e_metadata = TwoEnergyTool.high_e_viewer.projections.metadatas[0].metadata.copy()
        self.metadata["parent_metadata"] = self.parent_metadata
        self.metadata["high_e_metadata"] = self.high_e_metadata
        if "data_hierarchy_level" in self.parent_metadata:
            self.metadata["data_hierarchy_level"] = (
                self.parent_metadata["data_hierarchy_level"] + 1
            )
        else:
            self.metadata["data_hierarchy_level"] = 2
        self.table_label.value = "2E Metadata"

    def metadata_to_DataFrame(self):
        self.dataframe = None

    def create_metadata_box(self):

        self.metadata_vbox = VBox(
            [self.table_label],
            layout=Layout(align_items="center"),
        )

    def set_attributes_from_metadata(self, projections):
        pass


class Metadata_Align(Metadata):
    """
    Works with both Align and RunAlign instances.
    """

    def __init__(self):
        super().__init__()
        self.filename = "alignment_metadata.json"
        self.metadata["opts"] = {}
        self.metadata["methods"] = {}
        self.metadata["save_opts"] = {}
        self.table_label.value = "Alignment Metadata"

    def set_metadata(self, Align):
        self.metadata["metadata_type"] = "Align"
        self.metadata["opts"]["downsample"] = Align.downsample
        self.metadata["opts"]["ds_factor"] = int(Align.ds_factor)
        self.metadata["opts"]["pyramid_level"] = Align.pyramid_level
        self.metadata["opts"]["num_iter"] = Align.num_iter
        self.metadata["use_multiple_centers"] = Align.use_multiple_centers
        if self.metadata["use_multiple_centers"] and Align.Center.reg is not None:
            self.metadata["opts"]["center"] = Align.Center.reg_centers
        else:
            self.metadata["opts"]["center"] = Align.center
        self.metadata["opts"]["pad"] = (
            Align.padding_x,
            Align.padding_y,
        )
        self.metadata["opts"]["extra_options"] = Align.extra_options
        self.metadata["methods"] = Align.methods_opts
        self.metadata["save_opts"] = Align.save_opts
        self.metadata["px_range_x"] = Align.altered_viewer.px_range_x
        self.metadata["px_range_y"] = Align.altered_viewer.px_range_y
        self.metadata["parent_filedir"] = Align.projections.filedir
        self.metadata["parent_filename"] = Align.projections.filename
        self.metadata["copy_hists_from_parent"] = Align.copy_hists
        self.metadata["angles_rad"] = list(Align.projections.angles_rad)
        self.metadata["angles_deg"] = list(Align.projections.angles_deg)
        self.metadata["angle_start"] = Align.projections.angles_deg[0]
        self.metadata["angle_end"] = Align.projections.angles_deg[-1]
        self.set_metadata_obj_specific(Align)

    def set_metadata_obj_specific(self, Align):
        self.metadata["opts"][
            "shift_full_dataset_after"
        ] = Align.shift_full_dataset_after
        self.metadata["opts"]["upsample_factor"] = Align.upsample_factor
        self.metadata["opts"]["pre_alignment_iters"] = Align.pre_alignment_iters
        self.metadata["use_subset_correlation"] = Align.use_subset_correlation
        self.metadata["subset_x"] = Align.altered_viewer.subset_x
        self.metadata["subset_y"] = Align.altered_viewer.subset_y
        self.metadata["opts"]["num_batches"] = Align.num_batches

    def metadata_to_DataFrame(self):
        metadata_frame = {}
        time, title = parse_printed_time(self.metadata["analysis_time"])
        extra_headers = [
            "Prj X Range",
            "Prj Y Range",
            "Start Angle",
            "End Angle",
            title,
        ]
        metadata_frame["Headers"] = list(self.metadata["opts"].keys())
        center_idx = [
            i for i, key in enumerate(metadata_frame["Headers"]) if key == "center"
        ][0]
        metadata_frame["Headers"] = [
            metadata_frame["Headers"][i]
            .replace("_", " ")
            .title()
            .replace("Num", "No.")
            for i, key in enumerate(metadata_frame["Headers"]) if key != "pyramid_level"
        ]
        metadata_frame["Headers"] = metadata_frame["Headers"] + extra_headers
        extra_values = [
            self.metadata["px_range_x"],
            self.metadata["px_range_y"],
            self.metadata["angle_start"],
            self.metadata["angle_end"],
            time,
        ]
        extra_values = [str(extra_values[i]) for i in range(len(extra_values))]
        metadata_frame["Values"] = [
            str(self.metadata["opts"][key]) for key in self.metadata["opts"] if key != "pyramid_level"
        ] + extra_values
        if "use_multiple_centers" in self.metadata:
            if self.metadata["use_multiple_centers"]:
                metadata_frame["Values"][center_idx] = "Multiple"
        metadata_frame = {
            metadata_frame["Headers"][i]: metadata_frame["Values"][i]
            for i in range(len(metadata_frame["Headers"]))
        }
        sr = pd.Series(metadata_frame)
        df = pd.DataFrame(sr).transpose()
        s = df.style.hide(axis="index")
        s.set_table_styles(
            [
                {"selector": "th.col_heading", "props": "text-align: center;"},
                {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
                {"selector": "td", "props": "text-align: center;" "font-size: 1.2em; "},
                {
                    "selector": "th:not(.index_name)",
                    "props": "background-color: #0F52BA; color: white;",
                },
            ],
            overwrite=False,
        )

        self.dataframe = s

    def set_attributes_from_metadata(self, Align):
        Align.downsample = self.metadata["opts"]["downsample"]
        if "ds_factor" in self.metadata["opts"]:
            Align.ds_factor = self.metadata["opts"]["ds_factor"]
        if "downsample_factor" in self.metadata["opts"]:
            Align.ds_factor = self.metadata["opts"]["downsample_factor"]
        if "pyramid_level" in self.metadata["opts"]:
            Align.pyramid_level = self.metadata["opts"]["pyramid_level"]
        if "copy_hists_from_parent" in self.metadata:
            Align.copy_hists = self.metadata["copy_hists_from_parent"]
        Align.num_iter = self.metadata["opts"]["num_iter"]
        Align.center = self.metadata["opts"]["center"]
        (Align.padding_x, Align.padding_y) = self.metadata["opts"]["pad"]
        Align.pad = (Align.padding_x, Align.padding_y)
        Align.extra_options = self.metadata["opts"]["extra_options"]
        Align.methods_opts = self.metadata["methods"]
        Align.save_opts = self.metadata["save_opts"]
        if "use_multiple_centers" not in self.metadata:
            Align.use_multiple_centers = False
        else:
            Align.use_multiple_centers = self.metadata["use_multiple_centers"]
        if "px_range_x" in self.metadata.keys():
            Align.px_range_x = self.metadata["px_range_x"]
            Align.px_range_y = self.metadata["px_range_y"]
        else:
            Align.px_range_x = self.metadata["pixel_range_x"]
            Align.px_range_y = self.metadata["pixel_range_y"]
        self.set_attributes_object_specific(Align)

    def set_attributes_object_specific(self, Align):
        if "shift_full_dataset_after" in self.metadata["opts"]:
            Align.shift_full_dataset_after = self.metadata["opts"][
                "shift_full_dataset_after"
            ]
        Align.upsample_factor = self.metadata["opts"]["upsample_factor"]
        Align.pre_alignment_iters = self.metadata["opts"]["pre_alignment_iters"]
        Align.subset_x = self.metadata["subset_x"]
        Align.subset_y = self.metadata["subset_y"]
        Align.use_subset_correlation = self.metadata["use_subset_correlation"]
        Align.num_batches = self.metadata["opts"]["num_batches"]


class Metadata_Recon(Metadata_Align):
    def set_metadata(self, Recon):
        super().set_metadata(Recon)
        self.metadata["metadata_type"] = "Recon"
        self.filename = "recon_metadata.json"
        self.table_label.value = "Reconstruction Metadata"

    def set_metadata_obj_specific(self, Recon):
        pass

    def set_attributes_from_metadata(self, Recon):
        super().set_attributes_from_metadata(Recon)

    def set_attributes_object_specific(self, Recon):
        pass