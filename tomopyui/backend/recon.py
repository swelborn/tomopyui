import matplotlib.pyplot as plt
import datetime
import time
import json
import os
import tifffile as tf
import tomopy
import numpy as np

from time import perf_counter
from tomopy.recon import wrappers
from tomopy.misc.corr import circ_mask
from tomopy.recon import algorithm as tomopy_algorithm
from tomopyui.backend.align import TomoAlign
from tomopyui.backend.util.padding import *
from tomopyui._sharedvars import *
from tomopyui.backend.io import Metadata_Recon

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
        self.metadata = Metadata_Recon()
        self.metadata.set_metadata(Recon)
        self.metadata.set_attributes_from_metadata(self)
        if not self.downsample:
            self.downsample_factor = 1
        self.Recon = Recon
        self.import_metadata = self.Recon.projections.metadata
        self.projections = Recon.projections
        self.wd_parent = self.metadata.metadata["parent_filedir"]
        self.angles_rad = Recon.projections.angles_rad
        self.plot_output1 = Recon.plot_output1
        self.plot_output2 = Recon.plot_output2
        self.recon = None
        self.make_wd(suffix="recon")
        self.run()

    def save_data_after_obj_specific(self, savedir):
        self.metadata.filename = "recon_metadata.json"
        if self.metadata.metadata["save_opts"]["recon"]:
            if self.metadata.metadata["save_opts"]["npy"]:
                np.save(savedir / "recon", self.recon)
            if self.metadata.metadata["save_opts"]["tiff"]:
                tf.imwrite(savedir / "recon.tif", self.recon)
            if (
                not self.metadata.metadata["save_opts"]["tiff"]
                and not self.metadata.metadata["save_opts"]["npy"]
            ):
                tf.imwrite(savedir / "recon.tif", self.recon)
        self.Recon.run_list.append({savedir: self.metadata})

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

        # Initialization of reconstruction dataset
        tomo_shape = self.prjs.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )
        self.Recon.log.info("Starting" + method_str)
        self.Recon.log.info("Center of rotation:" + str(self.center))
        # TODO: parsing recon method could be done in an Align method
        if method_str == "SIRT_Plugin":
            self.recon = tomocupy_algorithm.recon_sirt_plugin(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                # rec=self.recon,
                center=self.center,
            )
        elif method_str == "SIRT_3D":
            self.recon = tomocupy_algorithm.recon_sirt_3D(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                # rec=self.recon,
                center=self.center,
            )
        elif method_str == "CGLS_3D":
            self.recon = tomocupy_algorithm.recon_cgls_3D_allgpu(
                self.prjs,
                self.angles_rad,
                num_iter=self.num_iter,
                # rec=self.recon,
                center=self.center,
            )
        elif self.current_recon_is_cuda:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": int(self.num_iter),
                # TODO: "extra_options": {},
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
            # defined in run.py
            os.environ["TOMOPY_PYTHON_THREADS"] = str(os.environ["num_cpu_cores"])
            if algorithm == "gridrec" or algorithm == "fbp":
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

        return self

    def run(self):
        """
        Reconstructs a TomoData object using options in GUI.
        """

        metadata_list = super().make_metadata_list()
        super().init_prj()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            tic = time.perf_counter()
            self.reconstruct()
            self.recon = unpad_rec_with_pad(self.recon, self.pad_ds)
            self.recon = circ_mask(self.recon, axis=0)
            toc = time.perf_counter()

            self.metadata.metadata["analysis_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_data_after(alignment=False)
