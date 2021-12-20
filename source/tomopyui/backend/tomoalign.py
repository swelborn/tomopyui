#!/usr/bin/env python

from copy import copy, deepcopy
from skimage.transform import rescale  # look for better option
from time import perf_counter
from tomocupy.prep.alignment import shift_prj_cp
from tomocupy.prep.alignment import align_joint
from .util.save_metadata import save_metadata
from .util.pad_projections import pad_projections
from .util.trim_padding import trim_padding

# from tqdm.notebook import tnrange, tqdm

import datetime
import json
import os
import tifffile as tf
import tomopyui.backend.tomodata as td
import numpy as np


class TomoAlign:
    """
    Class for performing alignments.

    Parameters
    ----------

    """

    def __init__(self, Align):

        self.Align = Align
        self.metadata = Align.metadata.copy()
        self.tomo = td.TomoData(metadata=Align.Import.metadata)
        if self.metadata["partial"]:
            self.prj_range_x = self.metadata["prj_range_x"]
            self.prj_range_y = self.metadata["prj_range_y"]
        self.shift = None
        self.sx = None
        self.sy = None
        self.conv = None
        self.recon = None
        self.pad = Align.metadata["opts"]["pad"]
        self.downsample = Align.metadata["opts"]["downsample"]
        self.num_batches = Align.metadata["opts"]["batch_size"]
        if self.downsample:
            self.downsample_factor = self.metadata["opts"]["downsample_factor"]
        else:
            self.downsample_factor = 1
        self.pad_ds = tuple([int(self.downsample_factor * x) for x in self.pad])
        self.center = self.Align.center * self.downsample_factor + self.pad_ds[0]
        self.wd_parent = Align.Import.wd
        self.plot_output1 = Align.plot_output1
        self.plot_output2 = Align.plot_output2
        self.make_wd()
        self._main()

    def make_wd(self):
        now = datetime.datetime.now()
        os.chdir(self.wd_parent)
        dt_string = now.strftime("%Y%m%d-%H%M-")
        try:
            os.mkdir(dt_string + "alignment")
            os.chdir(dt_string + "alignment")
        except:
            os.mkdir(dt_string + "alignment-1")
            os.chdir(dt_string + "alignment-1")
        save_metadata("overall_alignment_metadata.json", self.metadata)
        #!!!!!!!!!! make option for tiff file save
        if self.metadata["save_opts"]["tomo_before"]:
            np.save("projections_before_alignment", self.tomo.prj_imgs)
        self.wd = os.getcwd()

    def make_metadata_list(self):
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
        if self.metadata["partial"]:
            proj_range_x_low = self.prj_range_x[0]
            proj_range_x_high = self.prj_range_x[1]
            proj_range_y_low = self.prj_range_y[0]
            proj_range_y_high = self.prj_range_y[1]
            self.prj_for_alignment = self.tomo.prj_imgs[
                :,
                proj_range_y_low:proj_range_y_high:1,
                proj_range_x_low:proj_range_x_high:1,
            ].copy()
        else:
            self.prj_for_alignment = self.tomo.prj_imgs.copy()

        # Make it downsampled if desired.
        if self.downsample:
            # downsample images in stack
            self.prj_for_alignment = rescale(
                self.prj_for_alignment,
                (1, self.downsample_factor, self.downsample_factor),
                anti_aliasing=True,
            )
        # Pad
        self.prj_for_alignment, self.pad_ds = pad_projections(
            self.prj_for_alignment, self.pad_ds, 1
        )

    def align(self):
        """
        Aligns a TomoData object using options in GUI.
        """
        # This will contain more options later, as of now it only accepts
        # align_joint from tomocupy

        align_joint(self)

    def save_align_data(self):

        # if on the second alignment, go into the directory most recently saved
        # !!!!!!!!!!!! need change directory
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata["methods"].keys())[0]

        if (
            "SIRT_CUDA" in self.metadata["methods"]
            and "Faster" in self.metadata["methods"]["SIRT_CUDA"]
        ):
            if self.metadata["methods"]["SIRT_CUDA"]["Faster"]:
                method_str = method_str + "-faster"
            if self.metadata["methods"]["SIRT_CUDA"]["Fastest"]:
                method_str = method_str + "-fastest"
        os.mkdir(dt_string + method_str)
        os.chdir(dt_string + method_str)
        save_metadata("metadata.json", self.metadata)

        if self.metadata["save_opts"]["tomo_after"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("projections_after_alignment", self.tomo.prj_imgs)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("projections_after_alignment.tif", self.tomo.prj_imgs)
            if (
                not self.metadata["save_opts"]["tiff"]
                and not self.metadata["save_opts"]["npy"]
            ):
                tf.imwrite("projections_after_alignment.tif", self.tomo.prj_imgs)
        if self.metadata["save_opts"]["recon"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("last_recon", self.recon)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("last_recon.tif", self.recon)
            if (
                not self.metadata["save_opts"]["tiff"]
                and not self.metadata["save_opts"]["npy"]
            ):
                tf.imwrite("last_recon.tif", self.recon)

        np.save("sx", self.sx)
        np.save("sy", self.sy)
        np.save("conv", self.conv)

    def _main(self):
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
            new_prj_imgs = deepcopy(self.tomo.prj_imgs)
            new_prj_imgs, self.pad = pad_projections(new_prj_imgs, self.pad, 1)
            new_prj_imgs = shift_prj_cp(
                new_prj_imgs,
                self.sx,
                self.sy,
                self.num_batches,
                self.pad,
                use_corr_prj_gpu=False,
            )
            new_prj_imgs = trim_padding(new_prj_imgs)
            self.tomo = td.TomoData(
                prj_imgs=new_prj_imgs, metadata=self.Align.Import.metadata
            )
            toc = perf_counter()
            self.metadata["alignment_time"] = {
                "seconds": toc - tic,
                "minutes": (toc - tic) / 60,
                "hours": (toc - tic) / 3600,
            }
            self.save_align_data()
            # self.save_reconstructed_data()
