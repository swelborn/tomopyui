import datetime
import json
import os
import tifffile as tf
import numpy as np
import pathlib

from copy import copy, deepcopy
from time import perf_counter
from skimage.transform import rescale  # look for better option
from tomopy.prep.alignment import align_joint as align_joint_tomopy
from tomopyui.backend.util.padding import *
from tomopyui._sharedvars import *
from tomopyui.backend.io import Metadata_Align

# TODO: make this global
from tomopyui.widgets.helpers import import_module_set_env

cuda_import_dict = {"cupy": "cuda_enabled"}
import_module_set_env(cuda_import_dict)
if os.environ["cuda_enabled"] == "True":
    from ..tomocupy.prep.alignment import align_joint as align_joint_cupy
    from ..tomocupy.prep.alignment import shift_prj_cp


# TODO: create superclass for TomoRecon and TomoAlign, as they basically do the same thing.
class TomoAlign:
    """ """

    def __init__(self, Align):
        self.shift = None
        self.sx = None
        self.sy = None
        self.conv = None
        self.recon = None
        self.metadata = Metadata_Align()
        self.metadata.set_metadata(Align)
        self.metadata.set_attributes_from_metadata(self)
        if not self.downsample:
            self.downsample_factor = 1
        self.Align = Align
        self.projections = Align.projections
        self.data_before_align = deepcopy(Align.Import.projections.data)
        self.wd_parent = self.metadata.metadata["parent_filedir"]
        self.angles_rad = Align.projections.angles_rad
        self.plot_output1 = Align.plot_output1
        self.plot_output2 = Align.plot_output2
        self.make_wd()
        self.run()

    def make_wd(self, suffix="alignment"):
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y%m%d-%H%M-")
        dt_str = dt_str + suffix
        try:
            os.mkdir(self.wd_parent / dt_str)
        except Exception:
            dt_str = now.strftime("%Y%m%d-%H%M%S-")
            dt_str = dt_str + suffix
            os.mkdir(self.wd_parent / dt_str)
        self.metadata.filedir = pathlib.Path(self.wd_parent / dt_str)
        self.metadata.filename = "overall_" + suffix + "_metadata.json"
        if suffix == "alignment":
            self.metadata.metadata[
                "parent_metadata"
            ] = self.Align.Import.projections.metadatas[0].metadata.copy()
        else:
            self.metadata.metadata[
                "parent_metadata"
            ] = self.Recon.Import.projections.metadatas[0].metadata.copy()
        self.metadata.metadata["data_hierarchy_level"] = (
            self.metadata.metadata["parent_metadata"]["data_hierarchy_level"] + 1
        )
        self.metadata.save_metadata()
        # !!!!!!!!!! make option for tiff file save
        if self.metadata.metadata["save_opts"]["tomo_before"]:
            save_str = "projections_before_" + suffix
            save_str_tif = "projections_before_" + suffix + ".tif"
            if self.metadata.metadata["save_opts"]["npy"]:
                np.save(
                    self.metadata.filedir / save_str,
                    self.projections.data,
                )
            if self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(
                    self.metadata.filedir / save_str_tif,
                    self.projections.data,
                )
            if (
                not self.metadata.metadata["save_opts"]["tiff"]
                and not self.metadata.metadata["save_opts"]["npy"]
            ):
                tf.imwrite(
                    self.metadata.filedir / save_str_tif,
                    self.projections.data,
                )
        self.wd = self.metadata.filedir

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
            _ = Metadata_Align()
            _.metadata = self.metadata.metadata.copy()
            _.metadata["methods"] = _d
            newkey = key.replace(" ", "_")  # put underscores in method names
            if _.metadata["methods"][newkey]:
                metadata_list.append(_)  # append only true methods

        return metadata_list

    def init_prj(self):
        self.prjs = deepcopy(self.projections.data)
        self.prjs = self.prjs[
            :,
            self.pixel_range_y[0] : self.pixel_range_y[1],
            self.pixel_range_x[0] : self.pixel_range_x[1],
        ]
        # center of rotation change to fit new range
        self.center = self.center - self.pixel_range_x[0]
        self.pad_ds = tuple([int(self.downsample_factor * x) for x in self.pad])
        self.center = self.center + self.pad[0]

        # Downsample
        if self.downsample:
            self.prjs = rescale(
                self.prjs,
                (1, self.downsample_factor, self.downsample_factor),
                anti_aliasing=True,
            )
            # center of rotation change for downsampled data
            self.center = self.center * self.downsample_factor
            if self.use_subset_correlation:
                self.subset_x = [x * self.downsample_factor for x in self.subset_x]
                self.subset_y = [y * self.downsample_factor for y in self.subset_y]
        # Pad
        self.prjs, self.pad_ds = pad_projections(self.prjs, self.pad_ds)

    def align(self):
        """
        Aligns a TomoData object using options in GUI.
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
                import scipy.fft as fft

                fft.set_backend("scipy")
                if method == "gridrec" or method == "fbp":
                    self.prjs, self.sx, self.sy, self.conv = align_joint_tomopy(
                        self.prjs,
                        self.angles_rad,
                        upsample_factor=self.upsample_factor,
                        center=self.center,
                        algorithm=method,
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

    def save_data_after(self):
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata.metadata["methods"].keys())[0]
        savedir_str = dt_str + method_str
        savedir = self.wd / savedir_str
        os.mkdir(savedir)
        self.metadata.metadata["savedir"] = str(savedir)
        self.metadata.filedir = savedir
        self.save_data_after_obj_specific(savedir)
        self.metadata.save_metadata()

    def save_data_after_obj_specific(self, savedir):
        self.metadata.metadata["sx"] = list(self.sx)
        self.metadata.metadata["sy"] = list(self.sy)
        self.metadata.metadata["convergence"] = list(self.conv)
        self.metadata.filename = "alignment_metadata.json"
        if self.metadata.metadata["save_opts"]["tomo_after"]:
            if self.metadata.metadata["save_opts"]["npy"]:
                np.save(
                    savedir / "projections_after_alignment", self.projections_aligned
                )
            if self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(
                    savedir / "projections_after_alignment.tif",
                    self.projections_aligned,
                )

            # defaults to at least saving tiff if none are checked
            if (
                not self.metadata.metadata["save_opts"]["tiff"]
                and not self.metadata.metadata["save_opts"]["npy"]
            ):
                tf.imwrite(
                    savedir / "projections_after_alignment.tif",
                    self.projections_aligned,
                )
        if self.metadata.metadata["save_opts"]["recon"] and self.current_align_is_cuda:
            if self.metadata.metadata["save_opts"]["npy"]:
                np.save(savedir / "last_recon", self.recon)
            if self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(savedir / "last_recon.tif", self.recon)
            if (
                not self.metadata.metadata["save_opts"]["tiff"]
                and not self.metadata.metadata["save_opts"]["npy"]
            ):
                tf.imwrite(savedir / "last_recon.tif", self.recon)
        self.Align.run_list.append({str(savedir.stem): self.metadata})

    def _shift_prjs_after_alignment(self):
        new_prj_imgs = self.data_before_align
        new_prj_imgs, self.pad = pad_projections(new_prj_imgs, self.pad)
        if self.current_align_is_cuda:
            new_prj_imgs = shift_prj_cp(
                new_prj_imgs,
                self.sx,
                self.sy,
                self.num_batches,
                self.pad,
                use_corr_prj_gpu=False,
            )
        else:
            # TODO: make shift projections without cupy
            pass
        self.projections_aligned = trim_padding(new_prj_imgs)

    def run(self):
        """
        Reconstructs a TomoData object using options in GUI.
        """

        metadata_list = self.make_metadata_list()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            self.init_prj()
            tic = perf_counter()
            self.align()
            # make new dataset and pad/shift it
            self._shift_prjs_after_alignment()
            toc = perf_counter()
            self.metadata.metadata["analysis_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_data_after()
            # self.save_reconstructed_data()
