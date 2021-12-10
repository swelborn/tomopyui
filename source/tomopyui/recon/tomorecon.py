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
from tomopy.recon.rotation import find_center, find_center_vo

import tomopy.data.tomodata as td
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

    def __init__(self, metadata, callbacks=None, recon=None, tomo=None):

        self.metadata = metadata
        self.tomo = td.TomoData(metadata=self.metadata)
        if self.metadata["partial"]:
            self.prj_range_x = self.metadata["prj_range_x"]
            self.prj_range_y = self.metadata["prj_range_y"]
        self.recon = recon
        print(np.mean(self.tomo.prj_imgs))
        self.make_wd_and_go()
        self._main()

    def make_wd_and_go(self):
        now = datetime.datetime.now()
        os.chdir(self.metadata["fpath"])
        dt_string = now.strftime("%Y%m%d-%H%M-")
        os.mkdir(dt_string + "recon")
        os.chdir(dt_string + "recon")
        self.save_overall_metadata()
        self.recon_wd = os.getcwd()

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

    def make_prj_for_recon(self):
        # Take away part of it, if desired.
        if self.metadata["partial"]:
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

    def save_overall_metadata(self):
        # from https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("overall_recon_metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)

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

        # https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)

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

        os.chdir(self.recon_wd)

    def reconstruct(self):

        # Initialize variables from metadata for ease of reading:
        # ensure it only runs on 1 thread for CUDA
        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        num_iter = self.metadata["opts"]["num_iter"]
        init_tomo_shape = self.prj_for_recon.shape
        method_str = list(self.metadata["methods"].keys())[0]
        if method_str == "MLEM_CUDA":
            method_str = "EM_CUDA"
        # num_batches = self.metadata["opts"]["batch_size"]  # change to num_batches
        
        # Initialization of reconstruction dataset
        tomo_shape = self.prj_for_recon.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )

        # Options go into kwargs which go into recon()
        # center = find_center_vo(self.prj_for_recon)
        # print(center)
        center = init_tomo_shape[2]/2
        kwargs = {}
        options = {
            "proj_type": "cuda",
            "method": method_str,
            "num_iter": num_iter
        }
        kwargs["options"] = options
        print(np.mean(self.prj_for_recon))
        if (
            "SIRT_CUDA" in self.metadata["methods"]
            and "SIRT Plugin-Faster" in self.metadata["methods"]["SIRT_CUDA"]
        ):
            if self.metadata["methods"]["SIRT_CUDA"]["SIRT Plugin-Faster"] == True:
                self.recon = self.recon_sirt_3D(self.prj_for_recon, center)
                # self.recon = circ_mask(self.recon, 0)
            elif self.metadata["methods"]["SIRT_CUDA"]["SIRT 3D-Fastest"] == True:
                self.recon = self.recon_sirt_3D_allgpu(self.prj_for_recon, center)
                # self.recon = circ_mask(self.recon, 0)
        else:
            self.recon = algorithm.recon(
                self.prj_for_recon,
                self.tomo.theta,
                center=center,
                algorithm=wrappers.astra,
                ncore=None,
                init_recon=self.recon,
                **kwargs,
                )
            # self.recon = circ_mask(self.recon, 0)
            print(np.mean(self.recon))
            print(np.mean(self.prj_for_recon))
        return self

    def recon_sirt_3D(self, prj, center=None):
        # Init tomo in sinogram order
        sinograms = algorithm.init_tomo(prj, 0)
        num_proj = sinograms.shape[1]
        num_y = sinograms.shape[0]
        num_x = sinograms.shape[2]
        # assume angles used are the same as parent tomography
        angles = self.tomo.theta
        proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
        if center is not None:
            center_shift = -(center - num_x/2)
            proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
        vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
        projector = astra.create_projector("cuda3d", proj_geom, vol_geom)
        astra.plugin.register(astra.plugins.SIRTPlugin)
        W = astra.OpTomo(projector)
        rec_sirt = W.reconstruct("SIRT-PLUGIN", sinograms, self.metadata["opts"]["num_iter"])
        return rec_sirt

    def recon_sirt_3D_allgpu(self, prj, center=None):
        # Init tomo in sinogram order
        sinograms = algorithm.init_tomo(prj, 0)
        num_proj = sinograms.shape[1]
        num_y = sinograms.shape[0]
        num_x = sinograms.shape[2]
        # assume angles used are the same as parent tomography
        angles = self.tomo.theta
        # create projection geometry with shape of 
        proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
        # shifts the projection geometry so that it will reconstruct using the 
        # correct center.
        if center is not None:
            center_shift = -(center - num_x/2)
            proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
        vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
        sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
        rec_id = astra.data3d.create("-vol", vol_geom)
        reco_alg = "SIRT3D_CUDA"
        cfg = astra.astra_dict(reco_alg)
        cfg["ProjectionDataId"] = sinograms_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 2)
        rec_sirt = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinograms_id)
        return rec_sirt

    def _main(self):
        """
        Reconstructs a TomoData object using options in GUI.
        """

        self.make_prj_for_recon()
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
            # self.save_reconstructed_data()


def recon_sirt_3D_allgpu_static(prj, angles):
    # Init tomo in sinogram order
    sinograms = algorithm.init_tomo(prj, 0)
    num_proj = sinograms.shape[1]
    num_y = sinograms.shape[0]
    num_x = sinograms.shape[2]
    # assume angles used are the same as parent tomography
    # create projection geometry with shape of
    proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
    vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
    sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
    rec_id = astra.data3d.create("-vol", vol_geom)
    reco_alg = "SIRT3D_CUDA"
    cfg = astra.astra_dict(reco_alg)
    cfg["ProjectionDataId"] = sinograms_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, self.metadata["opts"]["num_iter"])
    rec_sirt = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinograms_id)
    return rec_sirt