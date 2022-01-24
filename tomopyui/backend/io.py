from abc import ABC, abstractmethod
import numpy as np
import pathlib
import tifffile as tf
import tomopy.prep.normalize as tomopy_normalize
import os
import dxchange
import re
from tomopy.sim.project import angles as angle_maker
import olefile
import pandas as pd


class IOBase:
    def __init__(self):

        self._data = np.random.rand(10, 100, 100)
        self._fullpath = None
        self.dtype = None
        self.shape = None
        self.pxX = self._data.shape[2]
        self.pxY = self._data.shape[1]
        self.pxZ = self._data.shape[0]
        self.size_gb = None
        self.folder = None
        self.filename = None
        self.extension = None
        self.parent = None
        self.energy = None
        self.raw = False
        self.single_file = False
        self.metadata = {}

    def _change_dtype(self, dtype):
        try:
            self._data = self._data.astype(dtype)
            self.dtype = dtype
        except TypeError:
            print("That's not a type accepted by numpy arrays")

    @property
    def data(self):
        return self._data

    @data.setter
    # set data info whenever setting data
    def data(self, value):
        (self.pxZ, self.pxY, self.pxX) = self._data.shape
        self.rangeX = (0, self.pxX - 1)
        self.rangeY = (0, self.pxY - 1)
        self.rangeZ = (0, self.pxZ - 1)
        self.size_gb = self._data.nbytes / 1073741824
        self.dtype = self._data.dtype
        self._data = value

    @property
    def fullpath(self):
        return self._fullpath

    @fullpath.setter
    def fullpath(self, value):
        self.folder = value.parent
        self.filename = value.name
        self.extension = value.suffix
        self._fullpath = value

    def _write_data_npy(self, folder: pathlib.Path, name: str):
        np.save(folder / name, self.data)

    def _write_data_tiff(self, folder: pathlib.Path, name: str):
        tf.imwrite(folder / name, self.data)

    def _file_finder(self, folder, filetypes: list):
        files = [pathlib.PurePath(f) for f in os.scandir(folder) if not f.is_dir()]
        files_with_ext = [
            file.name for file in files if any(x in file.name for x in filetypes)
        ]
        return files_with_ext


class ProjectionsBase(IOBase, ABC):
    # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
    aliases = {
        "prj_imgs": "data",
        "num_angles": "pxZ",
        "width": "pxX",
        "height": "pxY",
        "prj_range_x": "rangeX",
        "prj_range_y": "rangeY",
        "prj_range_z": "rangeZ",  # Could probably fix these
    }

    def __init__(self):
        super().__init__()
        self.angles_rad = None
        self.angles_deg = None
        self.allowed_extensions = [".npy", ".tiff", ".tif"]

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    @abstractmethod
    def import_metadata(self, folder):
        ...

    @abstractmethod
    def import_folder_projections(self, folder):
        ...

    @abstractmethod
    def import_file_projections(self, fullpath):
        ...

    @abstractmethod
    def set_options_from_frontend(self, Import):
        ...


class Projections_Prenormalized(ProjectionsBase):
    def import_metadata(self, fullpath):
        pass

    def import_folder_projections(self, folder):
        cwd = os.getcwd()
        os.chdir(folder)
        image_sequence = tf.TiffSequence()
        self.num_theta = len(image_sequence.files)
        self._data = image_sequence.asarray().astype(np.float32)
        self.data = self._data
        image_sequence.close()
        os.chdir(cwd)

    def import_file_projections(self, fullpath):

        if ".tif" in fullpath:
            # if there is a file name, checks to see if there are many more
            # tiffs in the folder. If there are, will upload all of them.
            filetypes = [".tif", ".tiff"]
            textfiles = self._file_finder(folder, filetypes)
            tiff_count_in_folder = len(textfiles)
            if tiff_count_in_folder > 50:
                self.import_folder_projections(fullpath.parent)
                pass
            self._data = dxchange.reader.read_tiff(fullpath).astype(np.float32)
            self._fullpath = fullpath
            self.fullpath = self._fullpath
            self.data = self._data

        elif ".npy" in fullpath:
            self._data = np.load(fullpath).astype(np.float32)
            self._fullpath = fullpath
            self.fullpath = self._fullpath
            self.data = self._data

    def set_options_from_frontend(self, Import):
        self.Import = Import
        self.folder = Import.fpath
        self.filename = Import.fname
        if self.filename is not "":
            self.fullpath = self.folder / self.filename

    def get_img_shape(self):

        if self.extension == ".tif" or self.extension == ".tiff":
            allowed_extensions = [".tiff", ".tif"]
            file_list = [
                pathlib.PurePath(f) for f in os.scandir(self.folder) if not f.is_dir()
            ]
            tiff_file_list = [
                file.name
                for file in file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            tiff_count_in_folder = len(tiff_file_list)
            with tf.TiffFile(self.fullpath) as tif:
                # if you select a file instead of a file path, it will try to
                # bring in the full folder
                if tiff_count_in_folder > 50:
                    sizeX = tif.pages[0].tags["ImageWidth"].value
                    sizeY = tif.pages[0].tags["ImageLength"].value
                    sizeZ = tiff_count_in_folder  # can maybe use this later
                else:
                    imagesize = tif.pages[0].tags["ImageDescription"]
                    size = json.loads(imagesize.value)["shape"]
                    sizeZ = size[0]
                    sizeY = size[1]
                    sizeX = size[2]

        elif self.extension == ".npy":
            size = np.load(self.fullpath, mmap_mode="r").shape
            sizeZ = size[0]
            sizeY = size[1]
            sizeX = size[2]

        return (sizeZ, sizeY, sizeX)

    def set_prj_ranges(self):
        self.prj_range_x = (0, self.prj_shape[2] - 1)
        self.prj_range_y = (0, self.prj_shape[1] - 1)
        self.prj_range_z = (0, self.prj_shape[0] - 1)


class RawProjectionsBase(ProjectionsBase, ABC):
    def __init__(self):
        super().__init__()
        self.flats = None
        self.flats_ind = None
        self.darks = None
        self.normalized = False

    def normalize_nf(self):
        self._data = tomopy_normalize.normalize_nf(
            self._data, self.flats, self.darks, self.flats_ind
        )
        self.data = self._data
        self.raw = False
        self.normalized = True

    def normalize(self):
        self._data = tomopy_normalize.normalize(self._data, self.flats, self.darks)
        self.data = self._data
        self.raw = False
        self.normalized = True

    @abstractmethod
    def import_metadata(self, folder):
        ...

    @abstractmethod
    def import_folder_all(self, folder):
        ...

    @abstractmethod
    def import_folder_projections(self, folder):
        ...

    @abstractmethod
    def import_folder_flats(self, folder):
        ...

    @abstractmethod
    def import_folder_darks(self, folder):
        ...

    @abstractmethod
    def import_file_all(self, fullpath):
        ...

    @abstractmethod
    def import_file_projections(self, fullpath):
        ...

    @abstractmethod
    def import_file_flats(self, fullpath):
        ...

    @abstractmethod
    def import_file_darks(self, fullpath):
        ...

    @abstractmethod
    def set_options_from_frontend(self, Import):
        ...


class RawRawProjectionsXRM_SSRL62(RawProjectionsBase):
    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True

    def import_metadata(self, folder):
        filetypes = [".txt"]
        textfiles = self._file_finder(folder, filetypes)
        scan_info_filepath = (
            folder / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filepath = (
            folder / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filepath)
        (
            self.flats_filenames,
            self.flats_ind,
            self.data_filenames,
        ) = self.separate_flats_projs()
        # assume that the first projection is the same as the rest for metadata
        self.metadata["PROJECTIONS"] = self.read_xrms_metadata([self.data_filenames[0]])
        if self.angles_from_filenames:
            self.get_angles_from_filenames()
        else:
            self.get_angles_from_metadata()
        self.pxZ = len(self.data_filenames)
        self.pxY = self.metadata["PROJECTIONS"][0]["image_height"]
        self.pxX = self.metadata["PROJECTIONS"][0]["image_width"]
        self.folder = folder

    def import_folder_all(self, folder):
        filetypes = [".txt"]
        textfiles = self._file_finder(folder, filetypes)
        scan_info_filepath = (
            folder / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filepath = (
            folder / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filepath)
        (
            self.flats_filenames,
            self.flats_ind,
            self.data_filenames,
        ) = self.separate_flats_projs()
        self.flats, self.metadata["FLATS"] = self.load_xrms(self.flats_filenames)
        self._data, self.metadata["PROJECTIONS"] = self.load_xrms(self.data_filenames)
        self.data = self._data
        self.darks = np.zeros_like(self.flats)
        if self.angles_from_filenames:
            self.get_angles_from_filenames()
        else:
            self.get_angles_from_metadata()
        self.folder = folder

    def import_folder_projections(self, folder):
        filetypes = [".txt"]
        textfiles = self._file_finder(folder, filetypes)
        scan_info_filepath = (
            folder / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filepath = (
            folder / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filepath)
        _, _, self.data_filenames = self.separate_flats_projs()
        self._data, self.metadata["PROJECTIONS"] = self.load_xrms(self.data_filenames)
        self.data = self._data
        self.folder = folder

    def import_folder_flats(self, folder):
        filetypes = [".txt"]
        textfiles = self._file_finder(folder, filetypes)
        scan_info_filepath = (
            folder / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filepath = (
            folder / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filepath)
        self.flats_filenames, self.flats_ind, _ = self.separate_flats_projs()
        self.flats, self.metadata["FLATS"] = self.load_xrms(self.flats_filenames)
        self.folder = folder

    def import_folder_darks(self, folder):
        pass

    def import_file_all(self, fullpath):
        pass

    def import_file_projections(self, fullpath):
        pass

    def import_file_flats(self, fullpath):
        pass

    def import_file_darks(self, fullpath):
        pass

    def set_options_from_frontend(self, Import):
        self.Import = Import
        self.folder = Import.fpath
        self.filename = Import.fname
        self.angles_from_filenames = Import.angles_from_filenames

    def parse_scan_info(self, scan_info):
        data_file_list = []
        self.metadata = []
        with open(scan_info, "r") as f:
            filecond = True
            for line in f.readlines():
                if "FILES" not in line and filecond:
                    self.metadata.append(line.strip())
                    filecond = True
                else:
                    filecond = False
                    _ = scan_info.parent / line.strip()
                    data_file_list.append(_)
        metadata_tp = map(self.string_num_totuple, self.metadata)
        self.metadata = {scanvar[0]: scanvar[1] for scanvar in metadata_tp}
        self.metadata["REFEVERYEXPOSURES"] = self.metadata["REFEVERYEXPOSURES"][1:]
        self.metadata = {key: int(self.metadata[key]) for key in self.metadata}
        self.metadata["FILES"] = data_file_list[1:]

    # def parse_run_script(self, txt_file):
    #     energies = [[]]
    #     flats = [[]]
    #     collects = [[]]
    #     with open(txt_file, "r") as f:
    #         for line in f.readlines():
    #             if line.startswith("sete "):
    #                 energies.append(float(line[5:]))
    #                 flats.append([])
    #                 collects.append([])
    #             elif line.startswith("collect "):
    #                 filename = line[8:].strip()
    #                 if "ref_" in filename:
    #                     flats[-1].append(Path(txt_file).parent / filename)
    #                 else:
    #                     collects[-1].append(Path(txt_file).parent / filename)
    #     return energies, flats, collects

    def separate_flats_projs(self):
        flats = [
            file.parent / file.name
            for file in self.metadata["FILES"]
            if "ref_" in file.name
        ]
        projs = [
            file.parent / file.name
            for file in self.metadata["FILES"]
            if "ref_" not in file.name
        ]
        ref_ind = [
            True if "ref_" in file.name else False for file in self.metadata["FILES"]
        ]
        ref_ind_positions = [i for i, val in enumerate(ref_ind) if val][
            :: self.metadata["REFNEXPOSURES"]
        ]
        num_ref_groups = len(ref_ind_positions)
        ref_ind = [
            j for j in ref_ind_positions for i in range(self.metadata["REFNEXPOSURES"])
        ]
        return flats, ref_ind, projs

    def get_angles_from_filenames(self):
        reg_exp = re.compile("_[+-]\d\d\d.\d\d")
        self.angles_deg = map(
            reg_exp.findall, [str(file) for file in self.data_filenames]
        )
        self.angles_deg = [float(angle[0][1:]) for angle in self.angles_deg]
        self.angles_rad = [x * np.pi / 180 for x in self.angles_deg]

    def get_angles_from_metadata(self):
        self.angles_rad = [
            filemetadata["thetas"][0] for filemetadata in self.metadata["PROJECTIONS"]
        ]
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]

    # Not using this, takes a long time to load metadata in
    def read_xrms_metadata(self, xrm_list):
        metadatas = []
        for i, filename in enumerate(xrm_list):
            ole = olefile.OleFileIO(str(filename))
            metadata = dxchange.reader.read_ole_metadata(ole)
            metadata.pop("reference_filename")
            metadata.pop("reference_data_type")
            metadata.pop("facility")
            metadata.pop("x-shifts")
            metadata.pop("y-shifts")
            metadata.pop("reference")
            metadatas.append(metadata)
        return metadatas

    def load_xrms(self, xrm_list):
        data_stack = None
        metadatas = []
        for i, filename in enumerate(xrm_list):
            data, metadata = dxchange.read_xrm(str(filename))
            metadata.pop("reference_filename")
            metadata.pop("reference_data_type")
            metadata.pop("facility")
            metadata.pop("x-shifts")
            metadata.pop("y-shifts")
            metadata.pop("reference")
            if data_stack is None:
                data_stack = np.zeros((len(xrm_list),) + data.shape, data.dtype)
            data_stack[i] = data
            metadatas.append(metadata)
            self.Import.upload_progress.value += 1
        data_stack = np.flip(data_stack, axis=1)
        return data_stack, metadatas

    def metadata_to_DataFrame(self):

        # change metadata keys to be better looking
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
        m = {keys[key]: self.metadata[key] for key in keys}

        if m["Order"] == 0:
            m["Order"] = "ABAB"
        else:
            m["Order"] = "ABBA"

        # create headers and data for table
        top_headers = []
        middle_headers = []
        middle_headers.append(["Energy", "Tomo", "Mosaic", "MultiExposure"])
        middle_headers.append(
            [
                "Repeat Scan",
                "Wait (s)",
                "Num. Exposures",
                "Images/Projection",
            ]
        )
        middle_headers.append(
            ["Num. Ref Exposures", "Ref/Num Exposures", "Order", "Ref Despeckle Avg"]
        )
        middle_headers.append(["Up", "Down", "Left", "Right"])
        top_headers.append(["Layers"])
        top_headers.append(["Image Information"])
        top_headers.append(["Acquisition Information"])
        top_headers.append(["Reference Information"])
        top_headers.append(["Mosaic Information"])
        data = [
            [m[key] for key in middle_headers[i]] for i in range(len(middle_headers))
        ]
        middle_headers.insert(1, ["X Pixels", "Y Pixels", "Num. θ"])
        data.insert(1, [self.pxX, self.pxY, self.pxZ])

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
        s = df.style.hide_index()
        s.set_table_styles(
            {
                ("Acquisition Information", "Repeat Scan"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Image Information", "X Pixels"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Reference Information", "Num. Ref Exposures"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
            },
            overwrite=False,
        )
        s.set_table_styles(
            {
                ("Mosaic Information", "Up"): [
                    {"selector": "td", "props": "border-left: 1px solid white"},
                    {"selector": "th", "props": "border-left: 1px solid white"},
                ]
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

        return s

    def string_num_totuple(self, s):
        return (
            "".join(c for c in s if c.isalpha()) or None,
            "".join(c for c in s if c.isdigit() or None),
        )
