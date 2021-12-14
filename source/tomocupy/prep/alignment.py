#!/usr/bin/env python

from tomopy.misc.corr import circ_mask
from tomopy.recon import wrappers
from cupyx.scipy import ndimage as ndi_cp
from skimage.registration import phase_cross_correlation
from tomopy.prep.alignment import scale as scale_tomo
from tomopy.recon import algorithm as tomopy_algorithm
from fastprogress.fastprogress import master_bar, progress_bar

import astra
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import tomocupy.recon.algorithm as tomocupy_algorithm


def align_joint(TomoAlign):
    
    # ensure it only runs on 1 thread for CUDA
    os.environ["TOMOPY_PYTHON_THREADS"] = "1"

    # Initialize variables from metadata for ease of reading:
    init_tomo_shape = TomoAlign.prj_for_alignment.shape
    num_iter = TomoAlign.metadata["opts"]["num_iter"]
    downsample = TomoAlign.metadata["opts"]["downsample"]            
    pad = TomoAlign.metadata["opts"]["pad"]
    method_str = list(TomoAlign.metadata["methods"].keys())[0]
    upsample_factor = TomoAlign.metadata["opts"]["upsample_factor"]
    num_batches = TomoAlign.metadata["opts"]["batch_size"] # change to num_batches
    center = TomoAlign.center

    # Needs scaling for skimage float operations.
    TomoAlign.prj_for_alignment, scl = scale_tomo(TomoAlign.prj_for_alignment)

    # Initialization of reconstruction dataset
    tomo_shape = TomoAlign.prj_for_alignment.shape
    TomoAlign.recon = np.empty(
        (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
    )

    # add progress bar for method. roughly a full-loop progress bar.
    # Initialize shift arrays
    TomoAlign.sx = np.zeros((init_tomo_shape[0]))
    TomoAlign.sy = np.zeros((init_tomo_shape[0]))
    TomoAlign.conv = np.zeros((num_iter))
    # start iterative alignment
    for n in range(num_iter):
        TomoAlign.Align.progress_shifting.value = 0
        TomoAlign.Align.progress_reprojection.value = 0
        TomoAlign.Align.progress_phase_cross_corr.value = 0
        _rec = TomoAlign.recon
        if TomoAlign.metadata["methods"]["SIRT_CUDA"]["SIRT Plugin-Faster"]:
            TomoAlign.recon = tomocupy_algorithm.recon_sirt_3D(
                TomoAlign.prj_for_alignment, 
                TomoAlign.tomo.theta,
                num_iter=1,
                rec=_rec,
                center=center)
        elif TomoAlign.metadata["methods"]["SIRT_CUDA"]["SIRT 3D-Fastest"]:
            TomoAlign.recon = tomocupy_algorithm.recon_sirt_3D_allgpu(
                TomoAlign.prj_for_alignment, 
                TomoAlign.tomo.theta, 
                num_iter=1,
                rec=_rec, 
                center=center)
        else:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": 1
                }
            kwargs["options"] = options

            TomoAlign.recon = tomopy_algorithm.recon(
                TomoAlign.prj_for_alignment,
                TomoAlign.tomo.theta,
                algorithm=wrappers.astra,
                init_recon=_rec,
                center=center,
                ncore=None,
                **kwargs,
            )
        TomoAlign.Align.progress_total.value = n + 1
        # update progress bar
        # method_bar.update()
        # break up reconstruction into batches along z axis
        TomoAlign.recon = np.array_split(TomoAlign.recon, num_batches, axis=0)
        # may not need a copy.
        _rec = TomoAlign.recon.copy()

        # initialize simulated projection cpu array
        sim = []

        # begin simulating projections using astra.
        # this could probably be made more efficient, right now I am not 
        # certain if I need to be deleting every time.
        
        simulate_projections(_rec, sim, center, TomoAlign.tomo.theta,
            progress=TomoAlign.Align.progress_reprojection)
        # del _rec
        sim = np.concatenate(sim, axis=1)
        # only flip the simulated datasets if using normal tomopy algorithm
        # can remove if it is flipped in the algorithm
        if (
            TomoAlign.metadata["methods"]["SIRT_CUDA"]["SIRT Plugin-Faster"] == False
            and TomoAlign.metadata["methods"]["SIRT_CUDA"]["SIRT 3D-Fastest"] == False
        ):
            sim = np.flip(sim, axis=0)

        # Cross correlation
        shift_cpu = []
        batch_cross_correlation(
            TomoAlign.prj_for_alignment,
            sim,
            shift_cpu,
            num_batches,
            upsample_factor,
            subset_correlation=False,
            blur=False,
            pad=TomoAlign.pad_ds,
            progress=TomoAlign.Align.progress_phase_cross_corr
        )
        TomoAlign.shift = np.concatenate(shift_cpu, axis=1)

        # Shifting.
        (TomoAlign.prj_for_alignment,
        TomoAlign.sx,
        TomoAlign.sy,
        TomoAlign.shift,
        err,
        TomoAlign.pad_ds,
        center) = shift_prj_update_shift_cp(
            TomoAlign.prj_for_alignment,
            TomoAlign.sx,
            TomoAlign.sy,
            TomoAlign.shift,
            num_batches,
            TomoAlign.pad_ds,
            center,
            downsample_factor=TomoAlign.downsample_factor,
            progress=TomoAlign.Align.progress_shifting
        )
        TomoAlign.conv[n] = np.linalg.norm(err)
        with TomoAlign.plot_output1:
            TomoAlign.plot_output1.clear_output(wait=True)
            plotIm(TomoAlign, sim)
            print(f"Error = {np.linalg.norm(err):3.3f}.")
        with TomoAlign.plot_output2:
            TomoAlign.plot_output2.clear_output(wait=True)
            plotSxSy(TomoAlign)

        TomoAlign.recon = np.concatenate(TomoAlign.recon, axis=0)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    # TomoAlign.recon = np.concatenate(TomoAlign.recon, axis=0)
    # Re-normalize data
    # method_bar.close()
    TomoAlign.prj_for_alignment *= scl
    TomoAlign.recon = circ_mask(TomoAlign.recon, 0)
    if downsample:
        TomoAlign.sx /= TomoAlign.downsample_factor
        TomoAlign.sy /= TomoAlign.downsample_factor
        TomoAlign.shift /= TomoAlign.downsample_factor
    
    TomoAlign.pad = tuple([x / TomoAlign.downsample_factor for x in TomoAlign.pad_ds])
    return TomoAlign

def plotIm(TomoAlign, sim, projection_num=50):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(TomoAlign.prj_for_alignment[projection_num], cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Projection Image")
    ax2.imshow(sim[projection_num], cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Re-projected Image")
    plt.show()

def plotSxSy(TomoAlign):
    plotrange = range(TomoAlign.prj_for_alignment.shape[0])
    figsxsy = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
    ax1.plot(plotrange, TomoAlign.sx/TomoAlign.downsample_factor)
    ax1.set_title("Sx")
    ax2.plot(plotrange, TomoAlign.sy/TomoAlign.downsample_factor)
    ax2.set_title("Sy")
    ax2.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
    plt.show()

def simulate_projections(rec, sim, center, theta, progress=None):
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
        if progress is not None: progress.value += 1

def batch_cross_correlation(prj, sim, shift_cpu, num_batches, upsample_factor, 
                        blur=True, rin=0.5, rout=0.8, subset_correlation=False,
                        mask_sim=True, pad=(0,0), progress=None):
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
        if progress is not None: progress.value += 1
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

def shift_prj_cp(prj, sx, sy, num_batches, pad, use_corr_prj_gpu=False):
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

def shift_prj_update_shift_cp(prj, sx, sy, shift, num_batches, pad, center, 
    downsample_factor=1, smart_shift=True, smart_pad=True, progress=None):
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
        if progress is not None: progress.value += 1

    # concatenate the final list and return
    prj_cpu = np.concatenate(prj_cpu, axis=0)
    err = np.concatenate(err)
    shifted_bool = np.concatenate(shifted_bool)
    sx = np.concatenate(sx, axis=0)
    sy = np.concatenate(sy, axis=0)
    shift = np.concatenate(shift, axis=1)
    return prj_cpu, sx, sy, shift, err, pad, center