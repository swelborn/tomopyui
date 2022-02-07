from abc import ABC, abstractmethod
import numpy as np
import pathlib
import tifffile as tf
import tomopy.prep.normalize as tomopy_normalize
import os
import json
import dxchange
import re
from tomopy.sim.project import angles as angle_maker
import olefile
import pandas as pd
from skimage.transform import rescale
import pathlib
from joblib import Parallel, delayed


class IOBase:
    def __init__(self):

        self._data = np.random.rand(10, 100, 100)
        self.imported = False
        self._fullpath = None
        self.dtype = None
        self.shape = None
        self.pxX = self._data.shape[2]
        self.pxY = self._data.shape[1]
        self.pxZ = self._data.shape[0]
        self.size_gb = None
        self.filedir = None
        self.filename = None
        self.extension = None
        self.parent = None
        self.energy = None
        self.raw = False
        self.single_file = False
        self.metadata = {}

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
        self.filedir = value.parent
        self.filename = value.name
        self.extension = value.suffix
        self._fullpath = value

    def _write_data_npy(self, filedir: pathlib.Path, name: str):
        np.save(filedir / name, self.data)

    def _check_downsampled_data(self):
        try:
            self.filedir_ds = pathlib.Path(self.filedir / "downsampled").mkdir(
                parents=True
            )
            print(self.filedir_ds)
            self.filedir_ds = pathlib.Path(self.filedir / "downsampled")
            print(self.filedir_ds)
        except FileExistsError:
            self.filedir_ds = pathlib.Path(self.filedir / "downsampled")
            try:
                self._load_ds_and_hists()
            except Exception:
                self._write_downsampled_data()
        else:
            self._write_downsampled_data()

    def _write_downsampled_data(self):
        ds_vals = [(1, 0.1, 0.1), (1, 0.25, 0.25), (1, 0.5, 0.5), (1, 0.75, 0.75)]
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in ds_vals]
        ds_data = Parallel(n_jobs=4)(delayed(rescale)(self.data, x) for x in ds_vals)
        ds_data.append(self.data)
        hists = Parallel(n_jobs=5)(delayed(np.histogram)(x, bins=100) for x in ds_data)
        hist_intensities = [hist[0] for hist in hists]
        bin_edges = [hist[1] for hist in hists]
        xvals = [[(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)] for b in bin_edges]
        for data, string in zip(ds_data, ds_vals_strs):
            np.save(self.filedir_ds / str("ds" + string), data)
        for hist_int, string, bin_edge, bin_center in zip(
            hist_intensities, ds_vals_strs, bin_edges, xvals
        ):
            np.savez(
                self.filedir_ds / str("ds" + string + "hist"),
                frequency=hist_int,
                edges=bin_edge,
                bin_centers=bin_center,
            )
        self._load_ds_and_hists()

    def _load_ds_and_hists(self):
        ds_vals = [(1, 0.1, 0.1), (1, 0.25, 0.25), (1, 0.5, 0.5), (1, 0.75, 0.75)]
        ds_vals_strs = [str(x[2]).replace(".", "p") for x in ds_vals]
        self.hists = [
            np.load(self.filedir_ds / str("ds" + string + "hist.npz"))
            for string in ds_vals_strs
        ]
        self.data_ds = [
            np.load(self.filedir_ds / str("ds" + string + ".npy"), mmap_mode="r")
            for string in ds_vals_strs
        ]

    def _write_data_tiff(self, filedir: pathlib.Path, name: str):
        tf.imwrite(filedir / name, self.data)

    def _file_finder(self, filedir, filetypes: list):
        files = [pathlib.PurePath(f) for f in os.scandir(filedir) if not f.is_dir()]
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
        "pixel_range_x": "rangeX",
        "pixel_range_y": "rangeY",
        "pixel_range_z": "rangeZ",  # Could probably fix these
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

    def save_normalized_as_npy(self):
        np.save(self.filedir / "normalized_projections.npy", self.data)

    @abstractmethod
    def import_metadata(self, filedir):
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
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

    def import_filedir_projections(self, filedir):
        cwd = os.getcwd()
        os.chdir(filedir)
        image_sequence = tf.TiffSequence()
        self._data = image_sequence.asarray().astype(np.float32)
        self.data = self._data
        image_sequence.close()
        self.make_angles()
        os.chdir(cwd)

    def import_file_projections(self, fullpath):

        if ".tif" in str(fullpath):
            # if there is a file name, checks to see if there are many more
            # tiffs in the filedir. If there are, will upload all of them.
            filetypes = [".tif", ".tiff"]
            textfiles = self._file_finder(fullpath.parent, filetypes)
            tiff_count_in_filedir = len(textfiles)
            if tiff_count_in_filedir > 50:
                self.import_filedir_projections(fullpath.parent)
                pass
            self._data = np.array(
                dxchange.reader.read_tiff(fullpath).astype(np.float32)
            )
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self._fullpath = fullpath
            self.fullpath = self._fullpath
            self.data = self._data
            self.make_angles()
            self.imported = True

        elif ".npy" in str(fullpath):
            self._data = np.load(fullpath).astype(np.float32)
            self._data = np.where(np.isfinite(self._data), self._data, 0)
            self._fullpath = fullpath
            self.fullpath = self._fullpath
            self.data = self._data
            self.make_angles()
            self.imported = True

    def set_options_from_frontend(self, Import, Uploader):
        self.angle_start = Import.angle_start
        self.angle_end = Import.angle_end
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        if self.filename != "":
            self.fullpath = self.filedir / self.filename

    def get_img_shape(self):

        if self.extension == ".tif" or self.extension == ".tiff":
            allowed_extensions = [".tiff", ".tif"]
            file_list = [
                pathlib.PurePath(f) for f in os.scandir(self.filedir) if not f.is_dir()
            ]
            tiff_file_list = [
                file.name
                for file in file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            tiff_count_in_filedir = len(tiff_file_list)
            with tf.TiffFile(self.fullpath) as tif:
                # if you select a file instead of a file path, it will try to
                # bring in the full filedir
                if tiff_count_in_filedir > 50:
                    sizeX = tif.pages[0].tags["ImageWidth"].value
                    sizeY = tif.pages[0].tags["ImageLength"].value
                    sizeZ = tiff_count_in_filedir  # can maybe use this later
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
        self.pixel_range_x = (0, self.prj_shape[2] - 1)
        self.pixel_range_y = (0, self.prj_shape[1] - 1)
        self.pixel_range_z = (0, self.prj_shape[0] - 1)

    def make_angles(self):
        self.angles_rad = angle_maker(
            self.pxZ,
            ang1=self.angle_start,
            ang2=self.angle_end,
        )
        self.angles_deg = [x * 180 / np.pi for x in self.angles_rad]


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
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    def normalize(self):
        self._data = tomopy_normalize.normalize(self._data, self.flats, self.darks)
        self._data = tomopy_normalize.minus_log(self._data)
        self.data = self._data
        self.raw = False
        self.normalized = True

    @abstractmethod
    def import_metadata(self, filedir):
        ...

    @abstractmethod
    def import_filedir_all(self, filedir):
        ...

    @abstractmethod
    def import_filedir_projections(self, filedir):
        ...

    @abstractmethod
    def import_filedir_flats(self, filedir):
        ...

    @abstractmethod
    def import_filedir_darks(self, filedir):
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


class RawProjectionsXRM_SSRL62(RawProjectionsBase):
    def __init__(self):
        super().__init__()
        self.allowed_extensions = self.allowed_extensions + [".xrm"]
        self.angles_from_filenames = True

    def import_metadata(self, filedir):
        filetypes = [".txt"]
        textfiles = self._file_finder(filedir, filetypes)
        scan_info_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filedir)
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
        self.filedir = filedir

    def import_filedir_all(self, filedir):
        filetypes = [".txt"]
        textfiles = self._file_finder(self.filedir, filetypes)
        scan_info_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filedir)
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
        self.filedir = filedir
        self.imported = True

    def import_filedir_projections(self, filedir):
        filetypes = [".txt"]
        textfiles = self._file_finder(filedir, filetypes)
        scan_info_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filedir)
        _, _, self.data_filenames = self.separate_flats_projs()
        self._data, self.metadata["PROJECTIONS"] = self.load_xrms(self.data_filenames)
        self.data = self._data
        self.filedir = filedir
        self.imported = True

    def import_filedir_flats(self, filedir):
        filetypes = [".txt"]
        textfiles = self._file_finder(filedir, filetypes)
        scan_info_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" in file][0]
        )
        run_script_filedir = (
            filedir / [file for file in textfiles if "ScanInfo" not in file][0]
        )
        self.parse_scan_info(scan_info_filedir)
        self.flats_filenames, self.flats_ind, _ = self.separate_flats_projs()
        self.flats, self.metadata["FLATS"] = self.load_xrms(self.flats_filenames)
        self.filedir = filedir

    def import_filedir_darks(self, filedir):
        pass

    def import_file_all(self, fullpath):
        pass

    def import_file_projections(self, fullpath):
        pass

    def import_file_flats(self, fullpath):
        pass

    def import_file_darks(self, fullpath):
        pass

    def set_options_from_frontend(self, Import, Uploader):
        self.filedir = Uploader.filedir
        self.filename = Uploader.filename
        self.angles_from_filenames = Import.angles_from_filenames
        self.upload_progress = Uploader.upload_progress

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
            self.upload_progress.value += 1
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


# https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
def safe_serialize(obj, f):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, f, default=default, indent=4)


def save_metadata(filename, metadata):
    with open(filename, "w+") as f:
        a = safe_serialize(metadata, f)


def load_metadata(filepath=None, filename=None, fullpath=None):
    if fullpath is not None:
        with open(fullpath) as f:
            metadata = json.load(f)
    else:
        fullpath = os.path.abspath(os.path.join(filepath, filename))
        with open(fullpath) as f:
            metadata = json.load(f)
    return metadata


def metadata_to_DataFrame(metadata):
    metadata_frame = {}
    time, title = parse_printed_time(metadata["analysis_time"])
    extra_headers = ["Prj X Range", "Prj Y Range", "Start Angle", "End Angle", title]
    metadata_frame["Headers"] = list(metadata["opts"].keys())
    metadata_frame["Headers"] = [
        metadata_frame["Headers"][i].replace("_", " ").title().replace("Num", "No.")
        for i in range(len(metadata_frame["Headers"]))
    ]
    metadata_frame["Headers"] = metadata_frame["Headers"] + extra_headers
    extra_values = [
        metadata["pixel_range_x"],
        metadata["pixel_range_y"],
        metadata["angle_start"],
        metadata["angle_end"],
        time,
    ]
    extra_values = [str(extra_values[i]) for i in range(len(extra_values))]
    metadata_frame["Values"] = [
        str(metadata["opts"][key]) for key in metadata["opts"]
    ] + extra_values
    metadata_frame = {
        metadata_frame["Headers"][i]: metadata_frame["Values"][i]
        for i in range(len(metadata_frame["Headers"]))
    }
    sr = pd.Series(metadata_frame)
    df = pd.DataFrame(sr).transpose()
    s = df.style.hide_index()
    s.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
            {"selector": "td", "props": "text-align: center;" "font-size: 1.2em; "},
        ],
        overwrite=False,
    )
    return s


def parse_printed_time(timedict):
    if timedict["hours"] < 1:
        if timedict["minutes"] < 1:
            time = timedict["seconds"]
            title = "Time (s)"
        else:
            time = timedict["minutes"]
            title = "Time (min)"
    else:
        time = timedict["hours"]
        title = "Time (h)"

    time = f"{time:.1f}"
    return time, title
