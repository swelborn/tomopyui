from joblib import Parallel, delayed
from time import process_time, perf_counter, sleep
from tomopy.recon import wrappers
from tomopy.prep.alignment import scale as scale_tomo
from contextlib import nullcontext
from tomopy.recon import algorithm
from tomopy.misc.corr import circ_mask
from tomopyui.backend.io import save_metadata, load_metadata
from tomopy.recon import algorithm as tomopy_algorithm
from tomopyui.backend.align import TomoAlign
from tomopyui.backend.util.padding import *
from tomopyui._sharedvars import *

import matplotlib.pyplot as plt
import datetime
import time
import json
import os
import tifffile as tf
import tomopy
import numpy as np

# TODO: make this global
from tomopyui.widgets.helpers import import_module_set_env

cuda_import_dict = {"cupy": "cuda_enabled"}
import_module_set_env(cuda_import_dict)
if os.environ["cuda_enabled"] == "True":
    import astra
    import tomopyui.tomocupy.recon.algorithm as tomocupy_algorithm
    import cupy as cp

# TODO: create superclass for TomoRecon and TomoAlign, as they basically do the same
# thing.


class TomoRecon(TomoAlign):
    """ """

    def __init__(self, Recon, Align=None):
        # -- Creating attributes for reconstruction calcs ---------------------
        self._set_attributes_from_frontend(Recon)
        self.metadata["parent_filedir"] = self.projections.filedir
        self.metadata["parent_filename"] = self.projections.filename
        self.metadata["angle_start"] = self.projections.angles_deg[0]
        self.metadata["angle_end"] = self.projections.angles_deg[-1]
        self.recon = None
        self.make_wd()
        self.run()

    def _set_attributes_from_frontend(self, Recon):
        # TODO: Not good to pass the whole object in. This is only passed in here for
        # updating progress bars. Probably can pass references.
        self.Recon = Recon
        self.center = Recon.center
        self.projections = Recon.projections
        self.angles_rad = Recon.projections.angles_rad
        self.wd_parent = Recon.projections.filedir
        self.metadata = Recon.metadata.copy()
        self.pixel_range_x = Recon.pixel_range_x
        self.pixel_range_y = Recon.pixel_range_y
        self.pad = (Recon.paddingX, Recon.paddingY)
        self.downsample = Recon.downsample
        if self.downsample:
            self.downsample_factor = Recon.downsample_factor
        else:
            self.downsample_factor = 1
        self.num_iter = Recon.num_iter

    def make_wd(self):
        now = datetime.datetime.now()
        os.chdir(self.wd_parent)
        dt_string = now.strftime("%Y%m%d-%H%M-")
        try:
            os.mkdir(dt_string + "recon")
            os.chdir(dt_string + "recon")
        except:
            os.mkdir(dt_string + "recon-1")
            os.chdir(dt_string + "recon-1")
        save_metadata("overall_recon_metadata.json", self.metadata)
        if self.metadata["save_opts"]["tomo_before"]:
            np.save("projections_before_alignment", self.projections.data)
        self.wd = os.getcwd()

    def save_reconstructed_data(self):
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata["methods"].keys())[0]
        os.chdir(self.wd)
        savedir = dt_string + method_str
        os.mkdir(savedir)
        os.chdir(savedir)
        self.metadata["savedir"] = os.getcwd()
        save_metadata("metadata.json", self.metadata)

        if self.metadata["save_opts"]["tomo_before"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("tomo", self.prjs)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("tomo.tif", self.prjs)
            if (
                not self.metadata["save_opts"]["tiff"]
                and not self.metadata["save_opts"]["npy"]
            ):
                tf.imwrite("tomo.tif", self.prjs)
        if self.metadata["save_opts"]["recon"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("recon", self.recon)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("recon.tif", self.recon)
            if (
                not self.metadata["save_opts"]["tiff"]
                and not self.metadata["save_opts"]["npy"]
            ):
                tf.imwrite("recon.tif", self.recon)
        self.Recon.run_list.append({savedir: self.metadata})

    def reconstruct(self):

        # ensure it only runs on 1 thread for CUDA
        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        method_str = list(self.metadata["methods"].keys())[0]
        if (
            method_str in astra_cuda_recon_algorithm_underscores
            and os.environ["cuda_enabled"] == "True"
        ):
            self.current_recon_is_cuda = True
        else:
            self.current_recon_is_cuda = False

        if method_str == "MLEM_CUDA":
            method_str = "EM_CUDA"

        # Initialization of reconstruction dataset
        tomo_shape = self.prjs.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )
        self.Recon.log.info("Starting" + method_str)

        # TODO: parsing recon method could be done in an Align method
        if method_str == "SIRT_Plugin":
            self.recon = tomocupy_algorithm.recon_sirt_plugin(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                rec=self.recon,
                center=self.center,
            )
        elif method_str == "SIRT_3D":
            self.recon = tomocupy_algorithm.recon_sirt_3D(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                rec=self.recon,
                center=self.center,
            )
        elif method_str == "CGLS_3D":
            self.recon = tomocupy_algorithm.recon_cgls_3D_allgpu(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                rec=self.recon,
                center=self.center,
            )
        elif self.current_recon_is_cuda:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": self.num_iter,
                # TODO: "extra_options": {},
            }
            kwargs["options"] = options
            self.recon = tomopy_algorithm.recon(
                self.prjs,
                self.angles_rad,
                algorithm=wrappers.astra,
                init_recon=self.recon,
                center=self.center,
                ncore=1,
                **kwargs,
            )
        else:
            # defined in run.py
            os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
            if algorithm == "gridrec" or algorithm == "fbp":

                self.recon = tomopy_algorithm.recon(
                    self.prjs,
                    self.angles_rad,
                    algorithm=method_str,
                    init_recon=self.recon,
                    center=self.center,
                )
            else:
                self.recon = tomopy_algorithm.recon(
                    self.prjs,
                    self.angles_rad,
                    algorithm=method_str,
                    init_recon=self.recon,
                    center=self.center,
                    num_iter=self.num_iter,
                )

        return self

    def run(self):
        """
        Reconstructs a TomoData object using options in GUI.
        """

        metadata_list = super().make_metadata_list()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            super().init_prj()
            tic = time.perf_counter()
            self.reconstruct()
            self.recon = unpad_rec_with_pad(self.recon, self.pad_ds)
            self.recon = circ_mask(self.recon, axis=0)
            toc = time.perf_counter()

            self.metadata["analysis_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_reconstructed_data()
