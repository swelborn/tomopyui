from tomopyui.backend.io import Metadata, RawProjectionsBase

class RawProjectionsXRM_SSRL62C(RawProjectionsBase):
    """
    Raw data import functions associated with SSRL 6-2c. If you specify a folder filled
    with raw XRMS, a ScanInfo file, and a run script, this will automatically import
    your data and save it in a subfolder corresponding to the energy.
    """

    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True
        self.metadata = Metadata_SSRL62C_Raw()

    def import_metadata(self, Uploader):
        self.metadata = Metadata_SSRL62C_Raw()
        self.data_hierarchy_level = 0
        filetypes = [".txt"]
        textfiles = self._file_finder(Uploader.filedir, filetypes)
        self.scan_info_path = [
            Uploader.filedir / file for file in textfiles if "ScanInfo" in file
        ][0]
        self.parse_scan_info()
        self.determine_scan_type()
        self.run_script_path = [
            Uploader.filedir / file for file in textfiles if "ScanInfo" not in file
        ]
        if len(self.run_script_path) == 1:
            self.run_script_path = self.run_script_path[0]
        elif len(self.run_script_path) > 1:
            for file in self.run_script_path:
                with open(file, "r") as f:
                    line = f.readline()
                    if line.startswith(";;"):
                        self.run_script_path = file
        self.angles_from_filenames = True
        if self.scan_info["REFEVERYEXPOSURES"] == 1 and self.scan_type == "ENERGY_TOMO":
            (
                self.flats_filenames,
                self.data_filenames,
            ) = self.get_all_data_filenames_filedir(Uploader.filedir)
            self.angles_from_filenames = False
            self.from_txrm = True
            self.from_xrm = False
        else:
            (
                self.flats_filenames,
                self.data_filenames,
            ) = self.get_all_data_filenames()
            self.txrm = False
            self.from_xrm = True
        # assume that the first projection is the same as the rest for metadata
        self.scan_info["PROJECTION_METADATA"] = self.read_xrms_metadata(
            [self.data_filenames[0]]
        )
        self.scan_info["FLAT_METADATA"] = self.read_xrms_metadata(
            [self.flats_filenames[0]]
        )
        if self.angles_from_filenames:
            self.get_angles_from_filenames()
        else:
            self.get_angles_from_txrm()
        self.pxZ = len(self.angles_rad)
        self.pxY = self.scan_info["PROJECTION_METADATA"][0]["image_height"]
        self.pxX = self.scan_info["PROJECTION_METADATA"][0]["image_width"]
        self.binning = self.scan_info["PROJECTION_METADATA"][0]["camera_binning"]
        self.raw_data_type = self.scan_info["PROJECTION_METADATA"][0]["data_type"]
        if self.raw_data_type == 5:
            self.raw_data_type = np.dtype(np.uint16)
        elif self.raw_data_type == 10:
            self.raw_data_type = np.dtype(np.float32)
        self.pixel_size_from_metadata = (
            self.scan_info["PROJECTION_METADATA"][0]["pixel_size"] * 1000
        )  # nm
        self.get_and_set_energies(Uploader)
        self.filedir = Uploader.filedir
        self.metadata.filedir = Uploader.filedir
        self.metadata.filename = "raw_metadata.json"
        self.metadata.filepath = self.filedir / "raw_metadata.json"
        self.metadata.set_metadata(self)
        self.metadata.save_metadata()

    def import_filedir_all(self, Uploader):
        self.import_metadata(Uploader)
        self.user_overwrite_energy = Uploader.user_overwrite_energy
        self.filedir = Uploader.filedir
        self.selected_energies = Uploader.energy_select_multiple.value
        if len(self.selected_energies) == 0:
            self.selected_energies = (Uploader.energy_select_multiple.options[0],)
            Uploader.energy_select_multiple.value = (
                Uploader.energy_select_multiple.options[0],
            )
        if self.from_xrm:
            self.import_from_run_script(Uploader)
        elif self.from_txrm:
            self.import_from_txrm(Uploader)
        self.imported = True

    def import_filedir_projections(self, filedir):
        pass

    def import_filedir_flats(self, filedir):
        pass

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, filepath):
        pass

    def import_file_projections(self, filepath):
        pass

    def import_file_flats(self, filepath):
        pass

    def import_file_darks(self, filepath):
        pass

    def parse_scan_info(self):
        data_file_list = []
        self.scan_info = []
        with open(self.scan_info_path, "r") as f:
            filecond = True
            for line in f.readlines():
                if "FILES" not in line and filecond:
                    self.scan_info.append(line.strip())
                    filecond = True
                else:
                    filecond = False
                    _ = self.scan_info_path.parent / line.strip()
                    data_file_list.append(_)
        metadata_tp = map(self.string_num_totuple, self.scan_info)
        self.scan_info = {scanvar[0]: scanvar[1] for scanvar in metadata_tp}
        self.scan_info["REFEVERYEXPOSURES"] = self.scan_info["REFEVERYEXPOSURES"][1:]
        self.scan_info = {key: int(self.scan_info[key]) for key in self.scan_info}
        self.scan_info["FILES"] = data_file_list[1:]

    def determine_scan_type(self):
        self.scan_order = [
            (k, self.scan_info[k])
            for k in ("TOMO", "ENERGY", "MOSAIC", "MULTIEXPOSURE")
            if self.scan_info[k] != 0
        ]
        self.scan_order = sorted(self.scan_order, key=lambda x: x[1])
        self.scan_type = [string for string, val in self.scan_order]
        self.scan_type = "_".join(self.scan_type)

    def get_and_set_energies(self, Uploader):
        self.energy_guessed = False
        energies = []
        with open(self.run_script_path, "r") as f:
            for line in f.readlines():
                if line.startswith("sete "):
                    energies.append(float(line[5:]))
        self.energies_list_float = sorted(list(set(energies)))
        if self.energies_list_float == []:
            self.energies_list_float = [
                self.est_en_from_px_size(self.pixel_size_from_metadata, self.binning)
            ]
            self.energy_guessed = True
        self.energies_list_str = [
            f"{energy:08.2f}" for energy in self.energies_list_float
        ]
        self.raw_pixel_sizes = [
            self.calculate_px_size(energy, self.binning)
            for energy in self.energies_list_float
        ]
        Uploader.energy_select_multiple.options = self.energies_list_str
        if len(self.energies_list_str) > 10:
            Uploader.energy_select_multiple.rows = 10
        else:
            Uploader.energy_select_multiple.rows = len(self.energies_list_str)
        if len(self.energies_list_str) == 1 and self.energy_guessed:
            Uploader.energy_select_multiple.disabled = True
            Uploader.energy_select_multiple.description = "Est. Energy (eV):"
            Uploader.energy_overwrite_textbox.disabled = False
        else:
            Uploader.energy_select_multiple.description = "Energies (eV):"
            Uploader.energy_select_multiple.disabled = False
            Uploader.energy_overwrite_textbox.disabled = True

    def calculate_px_size(self, energy, binning):
        """
        Calculates the pixel size based on the energy and binning.
        From Johanna's calibration.
        """
        pixel_size = 0.002039449 * energy - 0.792164997
        pixel_size = pixel_size * binning
        return pixel_size

    def est_en_from_px_size(self, pixel_size, binning):
        """
        Estimates the energy based on the pixel size. This is for plain TOMO data where
        the energy is not available. You should be able to overwrite
        this in the frontend if energy cannot be found.
        Inverse of calculate_px_size.
        """
        # From Johanna's calibration doc
        energy = (pixel_size / binning + 0.792164997) / 0.002039449
        return energy

    def get_all_data_filenames(self):
        """
        Grabs the flats and projections filenames from scan info.

        Returns
        -------
        flats: list of pathlib.Path
            All flat file names in self.scan_info["FILES"]
        projs: list of pathlib.Path
            All projection file names in self.scan_info["FILES"]
        """

        flats = [
            file.parent / file.name
            for file in self.scan_info["FILES"]
            if "ref_" in file.name
        ]
        projs = [
            file.parent / file.name
            for file in self.scan_info["FILES"]
            if "ref_" not in file.name
        ]
        return flats, projs

    def get_all_data_filenames_filedir(self, filedir):
        """
        Grabs the flats and projections filenames from scan info.

        Returns
        -------
        flats: list of pathlib.Path
            All flat file names in self.scan_info["FILES"]
        projs: list of pathlib.Path
            All projection file names in self.scan_info["FILES"]
        """
        txrm_files = self._file_finder(filedir, [".txrm"])
        xrm_files = self._file_finder(filedir, [".xrm"])
        txrm_files = [filedir / file for file in txrm_files]
        xrm_files = [filedir / file for file in xrm_files]
        if any(["ref_" in str(file) for file in txrm_files]):
            flats = [
                file.parent / file.name for file in txrm_files if "ref_" in file.name
            ]
        else:
            flats = [
                file.parent / file.name for file in xrm_files if "ref_" in file.name
            ]
        if any(["tomo_" in str(file) for file in txrm_files]):
            projs = [
                file.parent / file.name for file in txrm_files if "tomo_" in file.name
            ]
        else:
            projs = [
                file.parent / file.name for file in xrm_files if "tomo_" in file.name
            ]
        return flats, projs

    def get_angles_from_filenames(self):
        """
        Grabs the angles from the file names in scan_info.
        """
        reg_exp = re.compile("_[+-0]\d\d\d.\d\d")
        self.angles_deg = map(
            reg_exp.findall, [str(file) for file in self.data_filenames]
        )
        self.angles_deg = [float(angle[0][1:]) for angle in self.angles_deg]
        seen = set()
        result = []
        for item in self.angles_deg:
            if item not in seen:
                seen.add(item)
                result.append(item)
        self.angles_deg = result
        self.angles_rad = [x * np.pi / 180 for x in self.angles_deg]

    def get_angles_from_metadata(self):
        """
        Gets the angles from the raw image metadata.
        """
        self.angles_rad = [
            filemetadata["thetas"][0]
            for filemetadata in self.scan_info["PROJECTION_METADATA"]
        ]
        seen = set()
        result = []
        for item in self.angles_rad:
            if item not in seen:
                seen.add(item)
                result.append(item)
        self.angles_rad = result
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    def get_angles_from_txrm(self):
        """
        Gets the angles from the raw image metadata.
        """
        self.angles_rad = self.scan_info["PROJECTION_METADATA"][0]["thetas"]
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    def read_xrms_metadata(self, xrm_list):
        """
        Reads XRM files and snags the metadata from them.

        Parameters
        ----------
        xrm_list: list(pathlib.Path)
            list of XRMs to grab metadata from
        Returns
        -------
        metadatas: list(dict)
            List of metadata dicts for files in xrm_list
        """
        metadatas = []
        for i, filename in enumerate(xrm_list):
            ole = olefile.OleFileIO(str(filename))
            metadata = read_ole_metadata(ole)
            metadatas.append(metadata)
        return metadatas

    def load_xrms(self, xrm_list, Uploader):
        """
        Loads XRM data from a file list in order, concatenates them to produce a stack
        of data (npy).

        Parameters
        ----------
        xrm_list: list(pathlib.Path)
            list of XRMs to upload
        Uploader: `Uploader`
            Should have an upload_progress attribute. This is the progress bar.
        Returns
        -------
        data_stack: np.ndarray()
            Data grabbed from xrms in xrm_list
        metadatas: list(dict)
            List of metadata dicts for files in xrm_list
        """
        data_stack = None
        metadatas = []
        for i, filename in enumerate(xrm_list):
            data, metadata = read_xrm(str(filename))
            if data_stack is None:
                data_stack = np.zeros((len(xrm_list),) + data.shape, data.dtype)
            data_stack[i] = data
            metadatas.append(metadata)
            Uploader.upload_progress.value += 1
        data_stack = np.flip(data_stack, axis=1)
        return data_stack, metadatas

    def load_txrm(self, txrm_filepath):
        data, metadata = read_txrm(str(txrm_filepath))
        # rescale -- camera saturates at 4k -- can double check this number later.
        # should not impact reconstruction
        data = rescale_intensity(data, in_range=(0, 4096), out_range="dtype")
        data = img_as_float32(data)
        return data, metadata

    def import_from_txrm(self, Uploader):
        """
        Script to upload selected data from selected txrm energies.

        If an energy is selected on the frontend, it will be added to the queue to
        upload and normalize.

        This reads the run script in the folder. Each time "set e" is in the run script,
        this means that the energy is changing and signifies a new tomography.

        Parameters
        ----------
        Uploader: `Uploader`
            Should have an upload_progress, status_label, and progress_output attribute.
            This is for the progress bar and information during the upload progression.
        """
        parent_metadata = self.metadata.metadata.copy()
        if "data_hierarchy_level" not in parent_metadata:
            try:
                with open(self.filepath) as f:
                    parent_metadata = json.load(
                        self.run_script_path.parent / "raw_metadata.json"
                    )
            except Exception:
                pass
        for energy in self.selected_energies:
            _tmp_filedir = copy.deepcopy(self.filedir)
            self.metadata = Metadata_SSRL62C_Prenorm()
            self.metadata.set_parent_metadata(parent_metadata)
            Uploader.upload_progress.value = 0
            self.energy_str = energy
            self.energy_float = float(energy)
            self.px_size = self.calculate_px_size(float(energy), self.binning)
            Uploader.progress_output.clear_output()
            self.energy_label = Label(
                f"{energy} eV", layout=Layout(justify_content="center")
            )
            with Uploader.progress_output:
                display(self.energy_label)
            # Getting filename from specific energy
            self.flats_filename = [
                file.parent / file.name
                for file in self.flats_filenames
                if energy in file.name and "ref_" in file.name
            ]
            self.data_filename = [
                file.parent / file.name
                for file in self.data_filenames
                if energy in file.name and "tomo_" in file.name
            ]
            self.status_label = Label(
                "Uploading txrm.", layout=Layout(justify_content="center")
            )
            self.flats, self.scan_info["FLAT_METADATA"] = self.load_txrm(
                self.flats_filename[0]
            )
            self._data, self.scan_info["PROJECTION_METADATA"] = self.load_txrm(
                self.data_filename[0]
            )
            self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
            self.make_import_savedir(str(energy + "eV"))
            self.status_label.value = "Normalizing."
            self.normalize()
            self._data = np.flip(self._data, axis=1)
            # TODO: potentially do this in normalize, decide later
            # this removes negative values,
            self._data = self._data - np.median(self._data[self._data < 0])
            self._data[self._data < 0] = 0.0
            self.data = self._data
            self.status_label.value = "Calculating histogram of raw data and saving."
            self._np_hist_and_save_data()
            self.saved_as_tiff = False
            self.filedir = self.import_savedir
            if Uploader.save_tiff_on_import_checkbox.value:
                self.status_label.value = "Saving projections as .tiff."
                self.saved_as_tiff = True
                self.save_normalized_as_tiff()
            self.status_label.value = "Downsampling data."
            self._check_downsampled_data()
            self.status_label.value = "Saving metadata."
            self.data_hierarchy_level = 1
            self.metadata.set_metadata(self)
            self.metadata.filedir = self.import_savedir
            self.metadata.filename = "import_metadata.json"
            self.metadata.save_metadata()
            self.filedir = _tmp_filedir
            self._close_hdf_file()

    def import_from_run_script(self, Uploader):
        """
        Script to upload selected data from a run script.

        If an energy is selected on the frontend, it will be added to the queue to
        upload and normalize.

        This reads the run script in the folder. Each time "set e" is in the run script,
        this means that the energy is changing and signifies a new tomography.

        Parameters
        ----------
        Uploader: `Uploader`
            Should have an upload_progress, status_label, and progress_output attribute.
            This is for the progress bar and information during the upload progression.
        """
        all_collections = [[]]
        energies = [[self.selected_energies[0]]]
        parent_metadata = self.metadata.metadata.copy()
        if "data_hierarchy_level" not in parent_metadata:
            try:
                with open(self.filepath) as f:
                    parent_metadata = json.load(
                        self.run_script_path.parent / "raw_metadata.json"
                    )
            except Exception:
                pass
        with open(self.run_script_path, "r") as f:
            for line in f.readlines():
                if line.startswith("sete "):
                    energies.append(f"{float(line[5:]):08.2f}")
                    all_collections.append([])
                elif line.startswith("collect "):
                    filename = line[8:].strip()
                    all_collections[-1].append(self.run_script_path.parent / filename)
        if len(energies) > 1:
            energies.pop(0)
            all_collections.pop(0)
        else:
            energies = energies[0]

        for energy, collect in zip(energies, all_collections):
            if energy not in self.selected_energies:
                continue
            else:
                _tmp_filedir = copy.deepcopy(self.filedir)
                self.metadata = Metadata_SSRL62C_Prenorm()
                self.metadata.set_parent_metadata(parent_metadata)
                Uploader.upload_progress.value = 0
                self.energy_str = energy
                self.energy_float = float(energy)
                self.px_size = self.calculate_px_size(float(energy), self.binning)
                Uploader.progress_output.clear_output()
                self.energy_label = Label(
                    f"{energy} eV", layout=Layout(justify_content="center")
                )
                with Uploader.progress_output:
                    display(Uploader.upload_progress)
                    display(self.energy_label)
                # Getting filename from specific energy
                self.flats_filenames = [
                    file.parent / file.name for file in collect if "ref_" in file.name
                ]
                self.data_filenames = [
                    file.parent / file.name
                    for file in collect
                    if "ref_" not in file.name
                ]
                self.proj_ind = [
                    True if "ref_" not in file.name else False for file in collect
                ]
                self.status_label = Label(
                    "Uploading .xrms.", layout=Layout(justify_content="center")
                )
                with Uploader.progress_output:
                    display(self.status_label)
                # Uploading Data
                Uploader.upload_progress.max = len(self.flats_filenames) + len(
                    self.data_filenames
                )
                self.flats, self.scan_info["FLAT_METADATA"] = self.load_xrms(
                    self.flats_filenames, Uploader
                )
                self._data, self.scan_info["PROJECTION_METADATA"] = self.load_xrms(
                    self.data_filenames, Uploader
                )
                self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
                self.make_import_savedir(str(energy + "eV"))
                projs, flats, darks = self.setup_normalize()
                self.status_label.value = "Calculating flat positions."
                self.flats_ind_from_collect(collect)
                self.status_label.value = "Normalizing."
                self._data = RawProjectionsBase.normalize_and_average(
                    projs,
                    flats,
                    darks,
                    self.flats_ind,
                    self.scan_info["NEXPOSURES"],
                    status_label=self.status_label,
                    compute=False,
                )
                self.data = self._data
                self.status_label.value = (
                    "Calculating histogram of raw data and saving."
                )
                self._dask_hist_and_save_data()
                self.saved_as_tiff = False
                self.filedir = self.import_savedir
                if Uploader.save_tiff_on_import_checkbox.value:
                    self.status_label.value = "Saving projections as .tiff."
                    self.saved_as_tiff = True
                    self.save_normalized_as_tiff()
                self.status_label.value = "Downsampling data."
                self._check_downsampled_data()
                self.status_label.value = "Saving metadata."
                self.data_hierarchy_level = 1
                self.metadata.set_metadata(self)
                self.metadata.filedir = self.import_savedir
                self.metadata.filename = "import_metadata.json"
                self.metadata.save_metadata()
                self.filedir = _tmp_filedir
                self._close_hdf_file()

    def setup_normalize(self):
        """
        Function to lazy load flats and projections as npy, convert to chunked dask
        arrays for normalization.

        Returns
        -------
        projs: dask array
            Projections chunked by scan_info["NEXPOSURES"]
        flats: dask array
            References chunked by scan_info["REFNEXPOSURES"]
        darks: dask array
            Zeros array with the same image dimensions as flats
        """
        data_dict = {
            self.hdf_key_raw_flats: self.flats,
            self.hdf_key_raw_proj: self._data,
        }
        self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
        self.filepath = self.import_savedir / self.normalized_projections_hdf_key
        self._open_hdf_file_read_write()
        z_chunks_proj = self.scan_info["NEXPOSURES"]
        z_chunks_flats = self.scan_info["REFNEXPOSURES"]
        self.flats = None
        self._data = None

        self.flats = da.from_array(
            self.hdf_file[self.hdf_key_raw_flats],
            chunks=(z_chunks_flats, -1, -1),
        ).astype(np.float32)

        self._data = da.from_array(
            self.hdf_file[self.hdf_key_raw_proj],
            chunks=(z_chunks_proj, -1, -1),
        ).astype(np.float32)
        darks = da.from_array(self.darks, chunks=(-1, -1, -1)).astype(np.float32)
        projs = self._data
        flats = self.flats

        return projs, flats, darks

    def flats_ind_from_collect(self, collect):
        """
        Calculates where the flats indexes are based on the current "collect", which
        is a collection under each "set e" from the run script importer.

        This will set self.flats_ind for normalization.
        """
        copy_collect = collect.copy()
        i = 0
        for pos, file in enumerate(copy_collect):
            if "ref_" in file.name:
                if i == 0:
                    i = 1
                elif i == 1:
                    copy_collect[pos] = 1
            elif "ref_" not in file.name:
                i = 0
        copy_collect = [value for value in copy_collect if value != 1]
        ref_ind = [True if "ref_" in file.name else False for file in copy_collect]
        ref_ind = [i for i in range(len(ref_ind)) if ref_ind[i]]
        ref_ind = sorted(list(set(ref_ind)))
        ref_ind = [ind - i for i, ind in enumerate(ref_ind)]
        # These indexes are at the position of self.data_filenames that
        # STARTS the next round after the references are taken
        self.flats_ind = ref_ind

    def string_num_totuple(self, s):
        """
        Helper function for import_metadata. I forget what it does. :)
        """
        return (
            "".join(c for c in s if c.isalpha()) or None,
            "".join(c for c in s if c.isdigit() or None),
        )


class RawProjectionsTiff_SSRL62B(RawProjectionsBase):
    """
    Raw data import functions associated with SSRL 6-2c. If you specify a folder filled
    with raw XRMS, a ScanInfo file, and a run script, this will automatically import
    your data and save it in a subfolder corresponding to the energy.
    """

    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True
        self.metadata_projections = Metadata_SSRL62B_Raw_Projections()
        self.metadata_references = Metadata_SSRL62B_Raw_References()
        self.metadata = Metadata_SSRL62B_Raw(
            self.metadata_projections, self.metadata_references
        )

    def import_data(self, Uploader):
        self.import_metadata()
        self.metadata_projections.set_extra_metadata(Uploader)
        self.metadata_references.set_extra_metadata(Uploader)
        self.metadata.filedir = self.metadata_projections.filedir
        self.filedir = self.metadata.filedir
        self.metadata.filepath = self.metadata.filedir / self.metadata.filename
        self.metadata.save_metadata()
        save_filedir_name = str(self.metadata_projections.metadata["energy_str"] + "eV")
        self.import_savedir = self.metadata_projections.filedir / save_filedir_name
        self.make_import_savedir(save_filedir_name)
        self.import_filedir_projections(Uploader)
        self.import_filedir_flats(Uploader)
        self.filedir = self.import_savedir
        projs, flats, darks = self.setup_normalize(Uploader)
        Uploader.import_status_label.value = "Normalizing projections"
        self._data = self.normalize_no_locations_no_average(
            projs, flats, darks, compute=False
        )
        self.data = self._data
        hist, r, bins, percentile = self._dask_hist()
        grp = self.hdf_key_norm
        data_dict = {
            self.hdf_key_norm_proj: self.data,
            grp + self.hdf_key_bin_frequency: hist[0],
            grp + self.hdf_key_bin_edges: hist[1],
            grp + self.hdf_key_image_range: r,
            grp + self.hdf_key_percentile: percentile,
        }
        self.dask_data_to_h5(data_dict, savedir=self.import_savedir)
        self._dask_bin_centers(grp, write=True, savedir=self.import_savedir)
        Uploader.import_status_label.value = "Downsampling data in a pyramid"
        self.filedir = self.import_savedir
        self._check_downsampled_data(label=Uploader.import_status_label)
        self.metadata_projections.set_attributes_from_metadata(self)
        self.metadata_prenorm = Metadata_SSRL62B_Prenorm()
        self.metadata_prenorm.set_metadata(self)
        self.metadata_prenorm.metadata["parent_metadata"] = (
            self.metadata.metadata.copy()
        )
        if Uploader.save_tiff_on_import_checkbox.value:
            self.status_label.value = "Saving projections as .tiff."
            self.saved_as_tiff = True
            self.save_normalized_as_tiff()
            self.metadata["saved_as_tiff"] = projections.saved_as_tiff
        self.metadata_prenorm.filedir = self.filedir
        self.metadata_prenorm.filepath = self.filedir / self.metadata_prenorm.filename
        self.metadata_prenorm.save_metadata()

        self.hdf_file.close()

    def import_metadata(self):
        self.metadata = Metadata_SSRL62B_Raw(
            self.metadata_projections, self.metadata_references
        )

    def import_metadata_projections(self, Uploader):
        self.projections_filedir = Uploader.projections_metadata_filepath.parent
        self.metadata_projections = Metadata_SSRL62B_Raw_Projections()
        self.metadata_projections.filedir = (
            Uploader.projections_metadata_filepath.parent
        )
        self.metadata_projections.filename = Uploader.projections_metadata_filepath.name
        self.metadata_projections.filepath = Uploader.projections_metadata_filepath
        self.metadata_projections.parse_raw_metadata()
        self.metadata_projections.set_extra_metadata(Uploader)

    def import_metadata_references(self, Uploader):
        self.references_filedir = Uploader.references_metadata_filepath.parent
        self.metadata_references = Metadata_SSRL62B_Raw_References()
        self.metadata_references.filedir = Uploader.references_metadata_filepath.parent
        self.metadata_references.filename = Uploader.references_metadata_filepath.name
        self.metadata_references.filepath = Uploader.references_metadata_filepath
        self.metadata_references.parse_raw_metadata()
        self.metadata_references.set_extra_metadata(Uploader)

    def import_filedir_all(self, Uploader):
        pass

    def import_filedir_projections(self, Uploader):
        tifffiles = self.metadata_projections.metadata["filenames"]
        tifffiles = [self.projections_filedir / file for file in tifffiles]
        Uploader.upload_progress.value = 0
        Uploader.upload_progress.max = len(tifffiles)
        Uploader.import_status_label.value = "Uploading projections"
        Uploader.progress_output.clear_output()
        with Uploader.progress_output:
            display(
                VBox(
                    [Uploader.upload_progress, Uploader.import_status_label],
                    layout=Layout(justify_content="center", align_items="center"),
                )
            )

        arr = []
        for file in tifffiles:
            arr.append(tf.imread(file))
            Uploader.upload_progress.value += 1
        Uploader.import_status_label.value = "Converting to numpy array"
        arr = np.array(arr)
        arr = np.rot90(arr, axes=(1, 2))
        Uploader.import_status_label.value = "Converting to dask array"
        arr = da.from_array(arr, chunks={0: "auto", 1: -1, 2: -1})
        Uploader.import_status_label.value = "Saving in normalized_projections.hdf5"
        data_dict = {self.hdf_key_raw_proj: arr}
        da.to_hdf5(self.import_savedir / self.normalized_projections_hdf_key, data_dict)

    def import_filedir_flats(self, Uploader):
        tifffiles = self.metadata_references.metadata["filenames"]
        tifffiles = [self.metadata_references.filedir / file for file in tifffiles]
        Uploader.upload_progress.value = 0
        Uploader.upload_progress.max = len(tifffiles)
        Uploader.import_status_label.value = "Uploading references"
        arr = []
        for file in tifffiles:
            arr.append(tf.imread(file))
            Uploader.upload_progress.value += 1
        Uploader.import_status_label.value = "Converting to numpy array"
        arr = np.array(arr)
        arr = np.rot90(arr, axes=(1, 2))
        Uploader.import_status_label.value = "Converting to dask array"
        arr = da.from_array(arr, chunks={0: "auto", 1: -1, 2: -1})
        Uploader.import_status_label.value = "Saving in normalized_projections.hdf5"
        data_dict = {self.hdf_key_raw_flats: arr}
        da.to_hdf5(self.import_savedir / self.normalized_projections_hdf_key, data_dict)

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, filepath):
        pass

    def import_file_projections(self, filepath):
        pass

    def import_file_flats(self, filepath):
        pass

    def import_file_darks(self, filepath):
        pass

    def setup_normalize(self, Uploader):
        """
        Function to lazy load flats and projections as npy, convert to chunked dask
        arrays for normalization.

        Returns
        -------
        projs: dask array
            Projections chunked by scan_info["NEXPOSURES"]
        flats: dask array
            References chunked by scan_info["REFNEXPOSURES"]
        darks: dask array
            Zeros array with the same image dimensions as flats
        """
        self.flats = None
        self._data = None
        self.hdf_file = h5py.File(
            self.import_savedir / self.normalized_projections_hdf_key, "a"
        )
        self.flats = self.hdf_file[self.hdf_key_raw_flats]
        self._data = self.hdf_file[self.hdf_key_raw_proj]
        self.darks = np.zeros_like(self.flats[0])[np.newaxis, ...]
        projs = da.from_array(self._data).astype(np.float32)
        flats = da.from_array(self.flats).astype(np.float32)
        darks = da.from_array(self.darks).astype(np.float32)
        return projs, flats, darks


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
        self.metadata["pixel_size_from_scan_info"] = (
            projections.pixel_size_from_metadata
        )
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
        self.metadata["normalized_projections_filename"] = (
            projections.normalized_projections_hdf_key
        )
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
