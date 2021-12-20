from tqdm.notebook import tnrange, tqdm
from joblib import Parallel, delayed
from time import process_time, perf_counter, sleep
from skimage.registration import phase_cross_correlation
from skimage import transform
from tomopy.recon import wrappers
from tomopy.prep.alignment import scale as scale_tomo
from contextlib import nullcontext
from tomopy.recon import algorithm
from tomopy.misc.corr import circ_mask
from skimage.transform import rescale
from .util.save_metadata import save_metadata
from tomopy.recon import algorithm as tomopy_algorithm


import tomocupy.recon.algorithm as tomocupy_algorithm
import tomopyui.backend.tomodata as td
import matplotlib.pyplot as plt
import datetime
import time
import json
import astra
import os
import tifffile as tf
import cupy as cp
import tomopy
import numpy as np


class TomoRecon:
    """
    Class for performing reconstructions.

    Parameters
    ----------
    tomo : TomoData object.
        Normalize the raw tomography data with the TomoData class. Then,
        initialize this class with a TomoData object.
    metadata : metadata from setup in widget-based notebook.
    """

    def __init__(self, Recon):

        self.Recon = Recon
        self.metadata = Recon.metadata.copy()
        self.tomo = td.TomoData(metadata=Recon.Import.metadata)
        self.partial = Recon.partial
        if self.partial:
            self.prj_range_x = self.metadata["prj_range_x"]
            self.prj_range_y = self.metadata["prj_range_y"]
        self.recon = None
        self.downsample = Recon.downsample
        self.downsample_factor = Recon.downsample_factor
        self.center = self.Recon.center * self.downsample_factor  # add padding?
        self.wd_parent = Recon.Import.wd
        self.num_iter = Recon.num_iter
        self.make_wd()
        self._main()

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
        self.wd = os.getcwd()

    def recon_multiple(self):
        metadata_list = []
        for key in self.metadata["methods"]:
            d = self.metadata["methods"]
            keys_to_remove = set(self.metadata["methods"].keys())
            keys_to_remove.remove(key)
            _d = {k: d[k] for k in set(list(d.keys())) - keys_to_remove}
            _metadata = self.metadata.copy()
            _metadata["methods"] = _d
            metadata_list.append(_metadata)
        return metadata_list

    def init_prj(self):
        # Take away part of it, if desired.
        if self.partial:
            proj_range_x_low = self.prj_range_x[0]
            proj_range_x_high = self.prj_range_x[1]
            proj_range_y_low = self.prj_range_y[0]
            proj_range_y_high = self.prj_range_y[1]
            self.prj_for_recon = self.tomo.prj_imgs[
                :,
                proj_range_y_low:proj_range_y_high:1,
                proj_range_x_low:proj_range_x_high:1,
            ].copy()
        else:
            self.prj_for_recon = self.tomo.prj_imgs.copy()

        # Make it downsampled if desired.
        if self.metadata["opts"]["downsample"]:
            downsample_factor = self.metadata["opts"]["downsample_factor"]

            # downsample images in stack
            self.prj_for_recon = rescale(
                self.prj_for_recon,
                (1, downsample_factor, downsample_factor),
                anti_aliasing=True,
            )

    def save_reconstructed_data(self):
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata["methods"].keys())[0]

        if (
            "SIRT_CUDA" in self.metadata["methods"]
            and "SIRT Plugin-Faster" in self.metadata["methods"]["SIRT_CUDA"]
        ):
            if self.metadata["methods"]["SIRT_CUDA"]["SIRT Plugin-Faster"]:
                method_str = method_str + "-faster"
            if self.metadata["methods"]["SIRT_CUDA"]["SIRT 3D-Fastest"]:
                method_str = method_str + "-fastest"
        os.mkdir(dt_string + method_str)
        os.chdir(dt_string + method_str)
        save_metadata("metadata.json", self.metadata)

        if self.metadata["save_opts"]["tomo_before"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("tomo", self.tomo.prj_imgs)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("tomo.tif", self.tomo.prj_imgs)
            if (
                not self.metadata["save_opts"]["tiff"]
                and not self.metadata["save_opts"]["npy"]
            ):
                tf.imwrite("tomo.tif", self.tomo.prj_imgs)
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

        os.chdir(self.wd)

    def reconstruct(self):

        # Initialize variables from metadata for ease of reading:
        # ensure it only runs on 1 thread for CUDA
        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        num_iter = self.metadata["opts"]["num_iter"]
        method_str = list(self.metadata["methods"].keys())[0]
        if method_str == "MLEM_CUDA":
            method_str = "EM_CUDA"
        # num_batches = self.metadata["opts"]["batch_size"]  # change to num_batches

        # Initialization of reconstruction dataset
        tomo_shape = self.prj_for_recon.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )
        self.Recon.log.info("Starting" + method_str)
        # Options go into kwargs which go into recon()
        # center = find_center_vo(self.prj_for_recon)
        # print(center)
        center = self.center

        if (
            "SIRT_CUDA" in self.metadata["methods"]
            and "SIRT Plugin-Faster" in self.metadata["methods"]["SIRT_CUDA"]
        ):
            if self.metadata["methods"]["SIRT_CUDA"]["SIRT Plugin-Faster"]:
                self.recon = tomocupy_algorithm.recon_sirt_3D(
                    self.prj_for_recon,
                    self.tomo.theta,
                    num_iter=num_iter,
                    rec=self.recon,
                    center=center,
                )
            elif self.metadata["methods"]["SIRT_CUDA"]["SIRT 3D-Fastest"]:
                self.recon = tomocupy_algorithm.recon_sirt_3D_allgpu(
                    self.prj_for_recon,
                    self.tomo.theta,
                    num_iter=num_iter,
                    rec=self.recon,
                    center=center,
                )
        else:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {"proj_type": "cuda", "method": method_str, "num_iter": num_iter}
            kwargs["options"] = options

            self.recon = tomopy_algorithm.recon(
                self.prj_for_recon,
                self.tomo.theta,
                algorithm=wrappers.astra,
                init_recon=self.recon,
                center=center,
                ncore=None,
                **kwargs,
            )
        return self

    def _main(self):
        """
        Reconstructs a TomoData object using options in GUI.
        """

        self.init_prj()
        metadata_list = self.recon_multiple()
        for i in range(len(metadata_list)):
            self.metadata = metadata_list[i]
            tic = time.perf_counter()
            self.reconstruct()
            toc = time.perf_counter()

            self.metadata["reconstruction_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_reconstructed_data()
