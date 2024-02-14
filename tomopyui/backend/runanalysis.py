import datetime
import os
import multiprocessing
import pathlib
import time
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import tifffile as tf
from scipy.stats import linregress
from tomopy.misc.corr import circ_mask
from tomopy.prep.alignment import align_joint as align_joint_tomopy
from tomopy.recon import algorithm as tomopy_algorithm
from tomopy.prep.alignment import shift_images as shift_images_tomopy
from tomopy.recon import wrappers
from scipy.fft import set_backend
from tomopyui._sharedvars import astra_cuda_recon_algorithm_underscores
from tomopyui.backend.io import Metadata_Align, Metadata_Recon, Projections_Child
from tomopyui.backend.util.padding import *
from tomopyui._sharedvars import cuda_import_dict
from tomopyui.widgets.helpers import import_module_set_env

import_module_set_env(cuda_import_dict)
if os.environ["cuda_enabled"] == "True":
    import tomopyui.tomocupy.recon.algorithm as tomocupy_algorithm
    from tomopyui.widgets.prep import shift_projections

    from ..tomocupy.prep.alignment import align_joint as align_joint_cupy

os.environ["num_cpu_cores"] = str(multiprocessing.cpu_count())


class RunAnalysisBase(ABC):
    """
    Base class for alignment and reconstruction objects.
    """

    def __init__(self, analysis_parent):
        self.recon = None
        self.skip_mk_wd_subdir = False
        self.analysis_parent = analysis_parent
        self.parent_projections = analysis_parent.projections
        self.projections = Projections_Child(analysis_parent.projections)
        self.metadata.set_metadata(analysis_parent)
        self.metadata.set_attributes_from_metadata(self)
        self.wd_parent = self.metadata.metadata["parent_filedir"]
        self.metadata.metadata["parent_filedir"] = str(
            self.metadata.metadata["parent_filedir"]
        )
        self.plot_output1 = analysis_parent.plot_output1
        self.plot_output2 = analysis_parent.plot_output2
        self.angles_rad = analysis_parent.projections.angles_rad
        self.make_wd()
        self.save_overall_metadata()
        self.save_data_before_analysis()
        self.run()

    def make_wd(self):
        """
        Creates a save directory to put projections into.
        """
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y%m%d-%H%M-")
        dt_str = dt_str + self.savedir_suffix
        self.wd = self.wd_parent / dt_str
        if self.wd.exists():
            dt_str = now.strftime("%Y%m%d-%H%M%S-")
            dt_str = dt_str + self.savedir_suffix
            self.wd = self.wd_parent / dt_str
        self.wd.mkdir()

    def save_overall_metadata(self):
        self.metadata.filedir = pathlib.Path(self.wd)
        self.metadata.filename = "overall_" + self.savedir_suffix + "_metadata.json"
        self.metadata.metadata["parent_metadata"] = (
            self.analysis_parent.projections.metadatas[0].metadata.copy()
        )
        self.metadata.metadata["data_hierarchy_level"] = (
            self.metadata.metadata["parent_metadata"]["data_hierarchy_level"] + 1
        )
        self.metadata.save_metadata()

    def save_data_before_analysis(self):
        if self.metadata.metadata["save_opts"]["Projections Before Alignment"]:
            save_str = "projections_before_" + self.savedir_suffix
            save_str_tif = "projections_before_" + self.savedir_suffix + ".tif"
            tf.imwrite(
                self.metadata.filedir / save_str_tif,
                self.projections.data,
            )

    def make_metadata_list(self):
        """
        Creates a metadata list for all of the methods check-marked in the UI.
        This is put into the for loop in run. Each item in the list is a
        separate metadata dictionary.
        """
        metadata_list = []
        for key in self.metadata.metadata["methods"]:
            d = self.metadata.metadata["methods"]
            keys_to_remove = set(self.metadata.metadata["methods"].keys())
            keys_to_remove.remove(key)
            _d = {
                k.replace(" ", "_"): d[k] for k in set(list(d.keys())) - keys_to_remove
            }
            _ = self.metadata_class()
            _.metadata = self.metadata.metadata.copy()
            _.metadata["methods"] = _d
            newkey = key.replace(" ", "_")  # put underscores in method names
            if _.metadata["methods"][newkey]:
                metadata_list.append(_)  # append only true methods

        return metadata_list

    def init_projections(self):
        self.px_range = (self.px_range_x, self.px_range_y)
        if not self.downsample:
            self.pyramid_level = -1
        self.ds_factor = np.power(2, int(self.pyramid_level + 1))  # will be 1 for no ds
        self.px_range_x_ds = [
            int(np.around(x / self.ds_factor)) for x in self.px_range_x
        ]
        self.px_range_y_ds = [
            int(np.around(y / self.ds_factor)) for y in self.px_range_y
        ]
        self.px_range_ds = (self.px_range_x_ds, self.px_range_y_ds)
        if self.pyramid_level == -1:
            self.projections.get_parent_data_from_hdf(self.px_range_ds)
            self.prjs = self.projections.data
        else:
            self.projections.get_parent_data_ds_from_hdf(
                self.pyramid_level, self.px_range_ds
            )
            self.prjs = self.projections.data_ds

        # Pad
        self.pad_ds = tuple([int(np.around(x / self.ds_factor)) for x in self.pad])
        self.prjs = pad_projections(self.prjs, self.pad_ds)

        # center of rotation change to fit new range
        if not self.use_multiple_centers:
            self.center = self.center / self.ds_factor
            self.center = self.center - self.px_range_x_ds[0] + self.pad_ds[0]
        if self.use_multiple_centers:
            # get centers for padded/downsampled data. just for the
            # computation, not saved in metadata
            centers_ds = [
                x[0] / self.ds_factor
                for x in self.analysis_parent.Center.center_slice_list
            ]
            slices_ds = [
                int(np.around(x[1] / self.ds_factor))
                for x in self.analysis_parent.Center.center_slice_list
            ]
            try:
                linreg = linregress(slices_ds, centers_ds)
            except ValueError:
                self.center = self.center / self.ds_factor
                self.center = self.center - self.px_range_x_ds[0] + self.pad_ds[0]
                self.metadata.metadata["use_multiple_centers"] = False
                self.use_multiple_centers = False
            else:
                m, b = linreg.slope, linreg.intercept
                slices_pad = range(
                    self.px_range_y_ds[0] - self.pad_ds[1],
                    self.px_range_y_ds[1] + self.pad_ds[1],
                )
                self.center = [m * x + b for x in slices_pad]
                self.center = [
                    c - self.px_range_x_ds[0] + self.pad_ds[0] for c in self.center
                ]

    def _save_data_after(self):
        if not self.skip_mk_wd_subdir:
            self.make_wd_subdir()
        self.metadata.filedir = self.wd_subdir

    def make_wd_subdir(self):
        """
        Creates a save directory to put projections into.
        """
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata.metadata["methods"].keys())[0]
        dt_str = dt_str + method_str
        self.wd_subdir = self.wd / dt_str
        if self.wd_subdir.exists():
            dt_str = now.strftime("%Y%m%d-%H%M%S-")
            dt_str = dt_str + self.savedir_suffix
            self.wd_subdir = self.wd / dt_str
        self.wd_subdir.mkdir()

    @abstractmethod
    def run(self): ...


class RunRecon(RunAnalysisBase):
    """ """

    def __init__(self, Recon):
        self.recon = None
        self.metadata_class = Metadata_Recon
        self.metadata = self.metadata_class()
        self.savedir_suffix = "recon"
        super().__init__(Recon)

    def save_data_after(self):
        super()._save_data_after()
        if self.metadata.metadata["save_opts"]["Reconstruction"]:
            tf.imwrite(self.wd_subdir / "recon.tif", self.recon)
        self.analysis_parent.run_list.append({self.wd_subdir: self.metadata})
        self.metadata.save_metadata()

    def reconstruct(self):
        # ensure it only runs on 1 thread for CUDA
        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        method_str = list(self.metadata.metadata["methods"].keys())[0]
        if (
            method_str in astra_cuda_recon_algorithm_underscores
            and os.environ["cuda_enabled"] == "True"
        ):
            self.current_recon_is_cuda = True
        else:
            self.current_recon_is_cuda = False

        if method_str == "MLEM_CUDA":
            method_str = "EM_CUDA"

        # TODO: parsing recon method could be done in an Align method
        if method_str == "SIRT_Plugin":
            self.recon = tomocupy_algorithm.recon_sirt_plugin(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                center=self.center,
            )
        elif method_str == "SIRT_3D":
            self.recon = tomocupy_algorithm.recon_sirt_3D(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                center=self.center,
            )
        elif method_str == "CGLS_3D":
            self.recon = tomocupy_algorithm.recon_cgls_3D_allgpu(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                center=self.center,
            )
        elif self.current_recon_is_cuda:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": int(self.num_iter),
            }
            kwargs["options"] = options
            self.recon = tomopy_algorithm.recon(
                self.prjs,
                self.angles_rad,
                algorithm=wrappers.astra,
                center=self.center,
                ncore=1,
                **kwargs,
            )
        else:
            os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
            if method_str == "gridrec" or method_str == "fbp":
                self.recon = tomopy_algorithm.recon(
                    self.prjs,
                    self.angles_rad,
                    algorithm=method_str,
                    center=self.center,
                )
            else:
                self.recon = tomopy_algorithm.recon(
                    self.prjs,
                    self.angles_rad,
                    algorithm=method_str,
                    center=self.center,
                    num_iter=self.num_iter,
                )
        self.recon = unpad_rec_with_pad(self.recon, self.pad_ds)
        self.recon = circ_mask(self.recon, axis=0)
        return self

    def save_data_before_analysis(self):
        pass

    def run(self):
        super().init_projections()
        metadata_list = super().make_metadata_list()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            tic = time.perf_counter()
            self.reconstruct()
            self.projections.data = self.recon
            toc = time.perf_counter()
            self.metadata.metadata["analysis_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_data_after()


class RunAlign(RunAnalysisBase):
    """ """

    def __init__(self, Align):
        self.shift = None
        self.sx = None
        self.sy = None
        self.conv = None
        self.metadata_class = Metadata_Align
        self.metadata = self.metadata_class()
        self.savedir_suffix = "alignment"
        super().__init__(Align)

    def init_projections(self):
        super().init_projections()
        if self.use_subset_correlation:
            self.subset_x = [int(x / self.ds_factor) for x in self.subset_x]
            self.subset_y = [int(y / self.ds_factor) for y in self.subset_y]
            self.subset_x = [int(x) + self.pad_ds[0] for x in self.subset_x]
            self.subset_y = [int(y) + self.pad_ds[1] for y in self.subset_y]
        else:
            self.subset_x = None
            self.subset_y = None

    def align(self):
        """
        Aligns projections using options in GUI.
        """
        for method in self.metadata.metadata["methods"]:
            if (
                method in astra_cuda_recon_algorithm_underscores
                and os.environ["cuda_enabled"] == "True"
            ):
                self.current_align_is_cuda = True
                align_joint_cupy(self)
            else:
                self.current_align_is_cuda = False
                os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
                with set_backend("scipy"):
                    if method == "gridrec" or method == "fbp":
                        self.prjs, self.sx, self.sy, self.conv = align_joint_tomopy(
                            self.prjs,
                            self.angles_rad,
                            upsample_factor=self.upsample_factor,
                            center=self.center,
                            algorithm=method,
                            iters=self.num_iter,
                        )
                    else:
                        self.prjs, self.sx, self.sy, self.conv = align_joint_tomopy(
                            self.prjs,
                            self.angles_rad,
                            upsample_factor=self.upsample_factor,
                            center=self.center,
                            algorithm=method,
                            iters=self.num_iter,
                        )

    def _shift_prjs_after_alignment(self):
        if self.shift_full_dataset_after:
            self.projections.get_parent_data_from_hdf(None)
            if self.current_align_is_cuda:
                self.projections._data = shift_projections(
                    self.projections.data, self.sx, self.sy
                )
                self.projections.data = self.projections._data
            else:
                self.projections._data = shift_images_tomopy(
                    self.projections.data, self.sx, self.sy
                )
        else:
            self.projections._data = self.prjs
            self.projections.data = self.projections._data

    # def _copy_parent_hists(self):
    #     if self.copy_hists:
    #         self.projections.get_parent_hists(0)

    def save_data_after(self):
        super()._save_data_after()
        self.metadata.metadata["sx"] = list(self.sx)
        self.metadata.metadata["sy"] = list(self.sy)
        self.metadata.metadata["convergence"] = list(self.conv)
        self.saved_as_hdf = False
        if (
            self.metadata.metadata["save_opts"]["Projections After Alignment"]
            or self.analysis_parent.save_after_alignment
        ):
            if self.metadata.metadata["save_opts"]["hdf"]:
                self.projections.filepath = (
                    self.wd_subdir / "normalized_projections.hdf5"
                )
                data_dict = {self.projections.hdf_key_norm_proj: self.projections.data}
                self.projections.dask_data_to_h5(data_dict)
                self.saved_as_hdf = True
            elif self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(
                    self.wd_subdir / "normalized_projections.tif",
                    self.projections.data,
                )
            else:
                self.projections.filepath = (
                    self.wd_subdir / "normalized_projections.hdf5"
                )
                data_dict = {self.projections.hdf_key_norm_proj: self.projections.data}
                self.projections.dask_data_to_h5(data_dict)
        if (
            self.metadata.metadata["save_opts"]["Reconstruction"]
            and self.current_align_is_cuda
        ):
            if self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(self.wd_subdir / "recon.tif", self.recon)

        self.analysis_parent.run_list.append({str(self.wd_subdir.stem): self.metadata})
        self.projections.metadata = self.metadata
        self.metadata.save_metadata()

    def run(self):
        """ """

        metadata_list = self.make_metadata_list()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            self.init_projections()
            tic = perf_counter()
            self.align()
            # make new dataset and pad/shift it
            self._shift_prjs_after_alignment()
            # self._copy_parent_hists()
            toc = perf_counter()
            self.metadata.metadata["analysis_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_data_after()
            # self.save_reconstructed_data()
