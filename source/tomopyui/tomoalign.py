from tqdm.notebook import tnrange, tqdm
from joblib import Parallel, delayed
from time import process_time, perf_counter, sleep
from skimage.registration import phase_cross_correlation
#from skimage import transform
# removed because slower than ndi
from scipy import ndimage as ndi
from cupyx.scipy import ndimage as ndi_cp
from tomopy.recon import wrappers
from tomopy.prep.alignment import scale as scale_tomo
from contextlib import nullcontext
from tomopy.recon import algorithm
from skimage.transform import rescale
from tomopy.misc.corr import circ_mask
from copy import deepcopy, copy
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


class TomoAlign:
    """
    Class for performing alignments.

    Parameters
    ----------
    tomo : TomoData object.
        Normalize the raw tomography data with the TomoData class. Then,
        initialize this class with a TomoData object.
    metadata : metadata from setup in widget-based notebook.
    """

    def __init__(
        self,
        tomo,
        metadata,
        alignment_wd=None,
        alignment_wd_child=None,
        prj_aligned=None,
        shift=None,
        sx=None,
        sy=None,
        recon=None,
        callbacks=None
    ):

        self.tomo = tomo  # tomodata object
        self.metadata = metadata
        self.prj_range_x = metadata["opts"]["prj_range_x"]
        self.prj_range_y = metadata["opts"]["prj_range_y"]
        self.shift = shift
        self.sx = sx
        self.sy = sy
        self.conv = None
        self.recon = recon
        self.alignment_wd = alignment_wd
        self.alignment_wd_child = alignment_wd_child

        # setting up output callback context managers
        if callbacks is not None:
            if "methodoutput" in callbacks:
                self.method_bar_cm = callbacks["methodoutput"]
            else:
                self.method_bar_cm = nullcontext()
            if "output1" in callbacks:
                self.output1_cm = callbacks["output1"]
            else:
                self.output1_cm = nullcontext()
            if "output2" in callbacks:
                self.output2_cm = callbacks["output2"]
            else:
                self.output2_cm = nullcontext()
        else:
            self.method_bar_cm = nullcontext()
            self.output1_cm = nullcontext()
            self.output2_cm = nullcontext()

        # creates working directory based on time
        # creates multiple alignments based on
        if self.metadata["alignmultiple"] == True:
            self.make_wd_and_go()
            self.align_multiple()
        else:
            if self.alignment_wd is None:
                self.make_wd_and_go()
            self.align()

    def make_wd_and_go(self):
        now = datetime.datetime.now()
        os.chdir(self.metadata["generalmetadata"]["workingdirectorypath"])
        dt_string = now.strftime("%Y%m%d-%H%M-")
        os.mkdir(dt_string + "alignment")
        os.chdir(dt_string + "alignment")
        self.save_align_metadata()
        if self.metadata["save_opts"]["tomo_before"]:
            np.save("projections_before_alignment", self.tomo.prj_imgs)
        self.alignment_wd = os.getcwd()

    def align_multiple(self):

        metadata_list = []
        for key in self.metadata["methods"]:
            d = self.metadata["methods"]
            keys_to_remove = set(self.metadata["methods"].keys())
            keys_to_remove.remove(key)
            _d = {k: d[k] for k in set(list(d.keys())) - keys_to_remove}
            _metadata = self.metadata.copy()
            _metadata["methods"] = _d
            _metadata["alignmultiple"] = False
            metadata_list.append(_metadata)

        for metadata in metadata_list:
            self.callbacks["button"].description = (
                "Starting" + " " + list(metadata["methods"].keys())[0]
            )
            self.__init__(self.tomo, metadata, alignment_wd=self.alignment_wd)

    def align(self):
        """
        Aligns a TomoData object using options in GUI.
        """
        proj_range_x_low = self.metadata["opts"]["prj_range_x"][0]
        proj_range_x_high = self.metadata["opts"]["prj_range_x"][1]
        proj_range_y_low = self.metadata["opts"]["prj_range_y"][0]
        proj_range_y_high = self.metadata["opts"]["prj_range_y"][1]
        self.prj_aligned = self.tomo.prj_imgs[
            :,
            proj_range_y_low:proj_range_y_high:1,
            proj_range_x_low:proj_range_x_high:1,
        ].copy()

        tic = time.perf_counter()
        self.joint_astra_cupy()
        toc = time.perf_counter()

        self.metadata["alignment_time"] = {
            "seconds": toc - tic,
            "minutes": (toc - tic) / 60,
            "hours": (toc - tic) / 3600,
        }

        self.save_align_data()

    def save_align_metadata(self):
        # from https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("overall_alignment_metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)

    def save_align_data(self):

        # if on the second alignment, go into the directory most recently saved
        if self.metadata["align_number"] > 0:
            os.chdir(self.alignment_wd_child)
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

        # save child working directory for use in multiple alignments
        self.alignment_wd_child = os.getcwd()

        # https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)

        if self.metadata["save_opts"]["tomo_after"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("projections_after_alignment", self.tomo.prj_imgs)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("projections_after_alignment.tif", self.tomo.prj_imgs)
            if not self.metadata["save_opts"]["tiff"] and not self.metadata["save_opts"]["npy"]:
                tf.imwrite("projections_after_alignment.tif", self.tomo.prj_imgs)
        if self.metadata["save_opts"]["recon"]:
            if self.metadata["save_opts"]["npy"]:
                np.save("last_recon", self.recon)
            if self.metadata["save_opts"]["tiff"]:
                tf.imwrite("last_recon.tif", self.recon)
            if not self.metadata["save_opts"]["tiff"] and not self.metadata["save_opts"]["npy"]:
                tf.imwrite("last_recon.tif", self.recon)

        np.save("sx", self.sx)
        np.save("sy", self.sy)
        np.save("conv", self.conv)
        
        if self.metadata["align_number"] == 0:
            os.chdir(self.alignment_wd)
        else:
            os.chdir(self.alignment_wd_child)

    def joint_astra_cupy(
        self
    ):
        # Initialize variables from metadata for ease of reading:
        # ensure it only runs on 1 thread for CUDA
        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        num_iter = self.metadata["opts"]["num_iter"]
        init_tomo_shape = self.prj_aligned.shape
        downsample = self.metadata["opts"]["downsample"]            
        pad = self.metadata["opts"]["pad"]
        method_str = list(self.metadata["methods"].keys())[0]
        upsample_factor = self.metadata["opts"]["upsample_factor"]
        num_batches = self.metadata["opts"]["batch_size"] # change to num_batches

        # Needs scaling for skimage float operations.
        self.prj_aligned, scl = scale_tomo(self.prj_aligned)

        # pad sample after downsampling. this avoid uncessary allocation of 
        # memory to an already-large array if downsampled.

        if downsample:
            downsample_factor = self.metadata["opts"]["downsample_factor"]

            # downsample images in stack
            self.prj_aligned = rescale(
                self.prj_aligned,
                (1, downsample_factor, downsample_factor),
                anti_aliasing=True,
            )

            #!!!!!!!!!!!! TODO: add option for finding center or specifying
            center = find_center_vo(self.prj_aligned)
            print("found center with vo")
            print(center)
            # add downsampled padding to the edges of the sample
            pad_ds = tuple([int(downsample_factor*x) for x in pad])
            center = center + pad_ds[0]
            self.prj_aligned, pad_ds = pad_projections(self.prj_aligned, pad_ds, 1)
        else:
            downsample_factor = 1
            pad_ds = pad
            center = 197
            center = center + pad_ds[0]
            self.prj_aligned, pad_ds = pad_projections(self.prj_aligned, pad_ds, 1)
            
        # Initialization of reconstruction dataset
        tomo_shape = self.prj_aligned.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )


        # add progress bar for method. roughly a full-loop progress bar.
        # with self.method_bar_cm:
        #     method_bar = tqdm(
        #         total=num_iter,
        #         desc=options["method"],
        #         display=True,
        #     )

        # Initialize shift arrays
        self.sx = np.zeros((init_tomo_shape[0]))
        self.sy = np.zeros((init_tomo_shape[0]))
        self.conv = np.zeros((num_iter))

        # start iterative alignment
        for n in range(num_iter):

            _rec = self.recon

            if self.metadata["methods"]["SIRT_CUDA"]["Faster"] == True:
                self.recon = self.recon_sirt_3D(self.prj_aligned, center=center)
            elif self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == True:
                self.recon = self.recon_sirt_3D_allgpu(self.prj_aligned, _rec, center=center)
            else:

                # Options go into kwargs which go into recon()
                kwargs = {}
                options = {
                    "proj_type": "cuda",
                    "method": method_str,
                    "num_iter": 1
                    }
                kwargs["options"] = options

                self.recon = algorithm.recon(
                    self.prj_aligned,
                    self.tomo.theta,
                    algorithm=wrappers.astra,
                    init_recon=_rec,
                    center=center,
                    ncore=None,
                    **kwargs,
                )
            # update progress bar
            # method_bar.update()

            # break up reconstruction into batches along z axis
            self.recon = np.array_split(self.recon, num_batches, axis=0)
            # may not need a copy.
            _rec = self.recon.copy()
            
            # initialize simulated projection cpu array
            sim = []

            # begin simulating projections using astra.
            # this could probably be made more efficient, right now I am not 
            # certain if I need to be deleting every time.
            
            with self.output1_cm:
                self.output1_cm.clear_output()
                simulate_projections(_rec, sim, center, self.tomo.theta)
                # del _rec
                sim = np.concatenate(sim, axis=1)

                # only flip the simulated datasets if using normal tomopy algorithm
                # can remove if it is flipped in the algorithm
                if (
                    self.metadata["methods"]["SIRT_CUDA"]["Faster"] == False
                    and self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == False
                ):
                    sim = np.flip(sim, axis=0)

                # Cross correlation
                shift_cpu = []
                batch_cross_correlation(self.prj_aligned, 
                    sim, shift_cpu,
                    num_batches, upsample_factor, subset_correlation=False, blur=False, pad=pad_ds)
                self.shift = np.concatenate(shift_cpu, axis=1)

                # Shifting 
                (self.prj_aligned, 
                self.sx, 
                self.sy, 
                self.shift, 
                err, 
                pad_ds, 
                center) = warp_prj_shift_cp(
                    self.prj_aligned, 
                    self.sx, 
                    self.sy, 
                    self.shift, 
                    num_batches,
                    pad_ds,
                    center, 
                    downsample_factor=downsample_factor   
                )
                self.conv[n] = np.linalg.norm(err)
            with self.output2_cm:
                self.output2_cm.clear_output(wait=True)
                self.plotIm(sim)
                self.plotSxSy(downsample_factor)
                print(f"Error = {np.linalg.norm(err):3.3f}.")

            self.recon = np.concatenate(self.recon, axis=0)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        # self.recon = np.concatenate(self.recon, axis=0)
        # Re-normalize data
        # method_bar.close()
        self.prj_aligned *= scl
        self.recon = circ_mask(self.recon, 0)
        if downsample:
            self.sx = self.sx / downsample_factor
            self.sy = self.sy / downsample_factor
            self.shift = self.shift / downsample_factor
        
        pad = tuple([x / downsample_factor for x in pad_ds])
        # make new dataset and pad/shift it for the next round
        new_prj_imgs = deepcopy(self.tomo.prj_imgs)
        new_prj_imgs, pad = pad_projections(new_prj_imgs, pad, 1)
        new_prj_imgs = warp_prj_cp(new_prj_imgs, self.sx, self.sy, num_batches, pad, use_corr_prj_gpu=False)
        new_prj_imgs = trim_padding(new_prj_imgs)
        self.tomo = td.TomoData(
            prj_imgs=new_prj_imgs, metadata=self.metadata["importmetadata"]["tomo"]
        )
        return self

    def plotIm(self, sim, projection_num=50):
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(self.prj_aligned[projection_num], cmap="gray")
        ax1.set_axis_off()
        ax1.set_title("Projection Image")
        ax2.imshow(sim[projection_num], cmap="gray")
        ax2.set_axis_off()
        ax2.set_title("Re-projected Image")
        plt.show()

    def plotSxSy(self, downsample_factor):
        plotrange = range(self.prj_aligned.shape[0])
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax1.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
        ax1.plot(plotrange, self.sx/downsample_factor)
        ax1.set_title("Sx")
        ax2.plot(plotrange, self.sy/downsample_factor)
        ax2.set_title("Sy")
        ax2.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
        plt.show()

    def recon_sirt_3D(self, prj, center):
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

    def recon_sirt_3D_allgpu(self, prj, rec, center=None):
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
        rec_id = astra.data3d.create("-vol", vol_geom, rec)
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


def transform_parallel(prj, sx, sy, shift, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1] * metadata["opts"]["downsample_factor"]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] * metadata["opts"]["downsample_factor"]
    )

    def transform_algorithm(prj, shift, sx, sy, m):
        shiftm = shift[:, m]
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m] + shiftm[1]) < shift_x_condition
            and np.absolute(sy[m] + shiftm[0]) < shift_y_condition
        ):
            sx[m] += shiftm[1]
            sy[m] += shiftm[0]
            err[m] = np.sqrt(shiftm[0] * shiftm[0] + shiftm[1] * shiftm[1])

            # similarity transform shifts in (x, y)
            # tform = transform.SimilarityTransform(translation=(shiftm[1], shiftm[0]))
            # prj[m] = transform.warp(prj[m], tform, order=5)

            # found that ndi is much faster than the above warp
            # uses opposite convention
            shift_tuple = (shiftm[0], shiftm[1])
            shift_tuple = tuple([-1*x for x in shift_tuple])
            prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm)(prj, shift, sx, sy, m)
        for m in range(num_theta)
        # for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj, sx, sy, err


def warp_projections(prj, sx, sy, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] 
    )

    def transform_algorithm_warponly(prj, sx, sy, m):
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m]) < shift_x_condition
            and np.absolute(sy[m]) < shift_y_condition
        ):
            # similarity transform shifts in (x, y)
            # see above note for ndi switch
            # tform = transform.SimilarityTransform(translation=(sx[m], sy[m]))
            # prj[m] = transform.warp(prj[m], tform, order=5)

            shift_tuple = (sy[m], sx[m])
            shift_tuple = tuple([-1*x for x in shift_tuple])
            prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm_warponly)(prj, sx, sy, m)
        for m in range(num_theta)
        # for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj


def init_new_from_prior(prior_tomoalign, metadata):
    prj_imgs = deepcopy(prior_tomoalign.tomo.prj_imgs)
    new_tomo = td.TomoData(
        prj_imgs=prj_imgs, metadata=metadata["importmetadata"]["tomo"]
    )
    new_align_object = TomoAlign(
        new_tomo,
        metadata,
        alignment_wd=prior_tomoalign.alignment_wd,
        alignment_wd_child=prior_tomoalign.alignment_wd_child,
    )
    return new_align_object


def trim_padding(prj):
    # https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
    xs, ys, zs = np.where(prj > 1e-7)

    minxs = np.min(xs)
    maxxs = np.max(xs)
    minys = np.min(ys)
    maxys = np.max(ys)
    minzs = np.min(zs)
    maxzs = np.max(zs)

    # extract cube with extreme limits of where are the values != 0
    result = prj[minxs : maxxs + 1, minys : maxys + 1, minzs : maxzs + 1]
    # not sure why +1 here.

    return result


def simulate_projections(rec, sim, center, theta):
    for batch in range(len(rec)):
    # for batch in tnrange(len(rec), desc="Re-projection", leave=True):
        _rec = rec[batch]
        vol_geom = astra.create_vol_geom(
            _rec.shape[1], _rec.shape[1], _rec.shape[0]
        )
        phantom_id = astra.data3d.create("-vol", vol_geom, data=_rec)
        proj_geom = astra.create_proj_geom(
            "parallel3d",
            1,
            1,
            _rec.shape[0],
            _rec.shape[1],
            theta,
        )
        if center is not None:
            center_shift = -(center - _rec.shape[1]/2)
            proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
        projections_id, _sim = astra.creators.create_sino3d_gpu(
            phantom_id, proj_geom, vol_geom
        )
        _sim = _sim.swapaxes(0, 1)
        sim.append(_sim)
        astra.data3d.delete(projections_id)
        astra.data3d.delete(phantom_id)

def batch_cross_correlation(prj, sim, shift_cpu, num_batches, upsample_factor, 
                            blur=True, rin=0.5, rout=0.8, subset_correlation=False,
                            mask_sim=True, pad=(0,0)):
    # TODO: the sign convention for shifting is bad here. 
    # To fix this, change to 
    # shift_gpu = phase_cross_correlation(_sim_gpu, _prj_gpu...)
    # convention right now:
    # if _sim is down and to the right, the shift tuple will be (-, -) 
    # before going positive. 
    # split into arrays for batch.
    _prj = np.array_split(prj, num_batches, axis=0)
    _sim = np.array_split(sim, num_batches, axis=0)
    for batch in range(len(_prj)):
    # for batch in tnrange(len(_prj), desc="Cross-correlation", leave=True):
        # projection images have been shifted. mask also shifts.
        # apply the "moving" mask to the simulated projections
        # simulated projections have data outside of the mask.
        if subset_correlation:
             _prj_gpu = cp.array(_prj[batch]
                [
                    :,
                    2*pad[1]:-2*pad[1]:1,
                    2*pad[0]:-2*pad[0]:1
                ], 
                dtype=cp.float32
                )
             _sim_gpu = cp.array(_sim[batch]
                [
                    :,
                    2*pad[1]:-2*pad[1]:1,
                    2*pad[0]:-2*pad[0]:1
                ], 
                dtype=cp.float32
                )
        else:
            _prj_gpu = cp.array(_prj[batch], dtype=cp.float32)
            _sim_gpu = cp.array(_sim[batch], dtype=cp.float32)

        if mask_sim:
            _sim_gpu = cp.where(_prj_gpu < 1e-7, 0, _sim_gpu)

        if blur:
            _prj_gpu = blur_edges_cp(_prj_gpu, rin, rout)
            _sim_gpu = blur_edges_cp(_sim_gpu, rin, rout)

        # e.g. lets say sim is (-50, 0) wrt prj. This would correspond to
        # a shift of [+50, 0]
        # In the warping section, we have to now warp prj by (-50, 0), so the 
        # SAME sign of the shift value given here.
        shift_gpu = phase_cross_correlation(
            _sim_gpu,
            _prj_gpu,
            upsample_factor=upsample_factor,
            return_error=False,
        )
        shift_cpu.append(cp.asnumpy(shift_gpu))
    # shift_cpu = np.concatenate(shift_cpu, axis=1)

def blur_edges_cp(prj, low=0, high=0.8):
    """
    Blurs the edge of the projection images using cupy.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    low : scalar, optional
        Min ratio of the blurring frame to the image size.
    high : scalar, optional
        Max ratio of the blurring frame to the image size.

    Returns
    -------
    ndarray
        Edge-blurred 3D stack of projection images.
    """
    if type(prj) is np.ndarray:
        prj_gpu = cp.array(prj, dtype=cp.float32)
    else:
        prj_gpu = prj
    dx, dy, dz = prj_gpu.shape
    rows, cols = cp.mgrid[:dy, :dz]
    rad = cp.sqrt((rows - dy / 2) ** 2 + (cols - dz / 2) ** 2)
    mask = cp.zeros((dy, dz))
    rmin, rmax = low * rad.max(), high * rad.max()
    mask[rad < rmin] = 1
    mask[rad > rmax] = 0
    zone = cp.logical_and(rad >= rmin, rad <= rmax)
    mask[zone] = (rmax - rad[zone]) / (rmax - rmin)
    prj_gpu *= mask
    return prj_gpu




def warp_prj_shift_cp(prj, sx, sy, shift, num_batches, pad, center, 
    downsample_factor=1, smart_shift=True, smart_pad=True):
    # Why is the error calculated in such a strange way?
    # Will use the standard used in tomopy here, but think of different way to
    # calculate error.
    # TODO: add checks for sx, sy having the same dimension as prj
    #
    # If the shift starts to get larger than the padding in one direction,
    # shift it to the center of the sx values. This should help to avoid 
    average_sx = None
    average_sy = None
    if smart_shift:
        cond1 = sx.max() > 0.95*pad[0]
        cond2 = sy.max() > 0.95*pad[1]
        cond3 = np.absolute(sx.min()) > 0.95*pad[0]
        cond4 = np.absolute(sy.min()) > 0.95*pad[1]
        if cond1 or cond2 or cond3 or cond4:
            print("applying smart shift")
            print(f"sx max: {sx.max()}")    
            print(f"sx min: {sx.min()}")
            average_sx = (sx.max() + sx.min())/2
            average_sy = (sy.max() + sy.min())/2
            sx_smart_shift = average_sx*np.ones_like(sx)
            sy_smart_shift = average_sy*np.ones_like(sy)
            sx -= sx_smart_shift
            sy -= sy_smart_shift
            print(f"sx max after shift: {sx.max()}")
            print(f"sx min after shift: {sx.min()}")
            center = center + average_sx
            if smart_pad:
                if average_sx < 1 and cond1 and cond3:
                    extra_pad = tuple([0.2*pad[0], 0])
                    center = center + extra_pad[0]
                    pad = np.array(extra_pad) + np.array(pad)
                    prj, extra_pad = pad_projections(prj, extra_pad, 1)
                if average_sy < 1 and cond2 and cond4:
                    extra_pad = tuple([0, 0.2*pad[1]])
                    pad = np.array(extra_pad) + np.array(pad)
                    prj, extra_pad = pad_projections(prj, extra_pad, 1)



    num_theta = prj.shape[0]
    # TODO: why +1??
    err = np.zeros((num_theta + 1, 1))
    shifted_bool = np.zeros((num_theta + 1, 1))

    # split all arrays up into batches. 
    err = np.array_split(err, num_batches)
    prj_cpu = np.array_split(prj, num_batches, axis=0)
    sx = np.array_split(sx, num_batches, axis=0)
    sy = np.array_split(sy, num_batches, axis=0)
    shift = np.array_split(shift, num_batches, axis=1)
    shifted_bool = np.array_split(shifted_bool, num_batches, axis=0)
    for batch in range(len(prj_cpu)):
    # for batch in tnrange(len(prj_cpu), desc="Shifting", leave=True):
        _prj_gpu = cp.array(prj_cpu[batch], dtype=cp.float32)

        for image in range(_prj_gpu.shape[0]):
            # err calc before if - 
            err[batch][image] = np.sqrt(
                shift[batch][0,image] * shift[batch][0,image] + 
                shift[batch][1,image] * shift[batch][1,image]
            )
            if (
                np.absolute(sx[batch][image] + 
                    shift[batch][1,image]) < pad[0]
                and 
                np.absolute(sy[batch][image] + 
                    shift[batch][0,image]) < pad[1]
            ):
                shifted_bool[batch][image] = 1
                sx[batch][image] += shift[batch][1,image]
                sy[batch][image] += shift[batch][0,image]
                shift_tuple = (shift[batch][0,image], shift[batch][1,image])
                _prj_gpu[image] = ndi_cp.shift(_prj_gpu[image], shift_tuple, order=5)

        prj_cpu[batch] = cp.asnumpy(_prj_gpu)

    # concatenate the final list and return
    prj_cpu = np.concatenate(prj_cpu, axis=0)
    err = np.concatenate(err)
    shifted_bool = np.concatenate(shifted_bool)
    sx = np.concatenate(sx, axis=0)
    sy = np.concatenate(sy, axis=0)
    shift = np.concatenate(shift, axis=1)
    return prj_cpu, sx, sy, shift, err, pad, center

def warp_prj_cp(prj, sx, sy, num_batches, pad, use_corr_prj_gpu=False):
    # add checks for sx, sy having the same dimension as prj
    prj_cpu = np.array_split(prj, num_batches, axis=0)
    _sx = np.array_split(sx, num_batches, axis=0)
    _sy = np.array_split(sy, num_batches, axis=0)
    for batch in range(len(prj_cpu)):
    # for batch in tnrange(len(prj_cpu), desc="Shifting", leave=True):
        _prj_gpu = cp.array(prj_cpu[batch], dtype=cp.float32)
        num_theta = _prj_gpu.shape[0]
        shift_y_condition = (
            pad[1]
        )
        shift_x_condition = (
            pad[0]
        )

        for image in range(_prj_gpu.shape[0]):  
            if (
                np.absolute(_sx[batch][image]) < shift_x_condition
                and np.absolute(_sy[batch][image]) < shift_y_condition
            ):
                shift_tuple = (_sy[batch][image], _sx[batch][image])
                _prj_gpu[image] = ndi_cp.shift(_prj_gpu[image], shift_tuple, order=5)

        prj_cpu[batch] = cp.asnumpy(_prj_gpu)
    prj_cpu = np.concatenate(prj_cpu, axis=0)
    return prj_cpu

# warning I don't think I fixed sign convention here.
def transform_parallel(prj, sx, sy, shift, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1] * metadata["opts"]["downsample_factor"]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] * metadata["opts"]["downsample_factor"]
    )

    def transform_algorithm(prj, shift, sx, sy, m):
        shiftm = shift[:, m]
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m] + shiftm[1]) < shift_x_condition
            and np.absolute(sy[m] + shiftm[0]) < shift_y_condition
        ):
            sx[m] += shiftm[1]
            sy[m] += shiftm[0]
            err[m] = np.sqrt(shiftm[0] * shiftm[0] + shiftm[1] * shiftm[1])

            # similarity transform shifts in (x, y)
            # tform = transform.SimilarityTransform(translation=(shiftm[1], shiftm[0]))
            # prj[m] = transform.warp(prj[m], tform, order=5)

            # found that ndi is much faster than the above warp
            # uses opposite convention
            shift_tuple = (shiftm[0], shiftm[1])
            shift_tuple = tuple([-1*x for x in shift_tuple])
            prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm)(prj, shift, sx, sy, m)
        for m in range(num_theta)
        # for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj, sx, sy, err

def pad_projections(prj, pad, downsample_factor):
    pad_ds = tuple([int(downsample_factor*x) for x in pad])
    npad_ds = ((0, 0), (pad_ds[1], pad_ds[1]), (pad_ds[0], pad_ds[0]))
    prj = np.pad(
        prj, npad_ds, mode="constant", constant_values=0
    )
    return prj, pad_ds