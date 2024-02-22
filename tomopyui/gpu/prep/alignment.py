import os

import astra
import bqplot as bq
import cupy as cp
import numpy as np
from bqplot_image_gl import ImageGL
from cupyx.scipy import ndimage as ndi_cp
from ipywidgets import *
from tomopy.misc.corr import circ_mask
from tomopy.prep.alignment import scale as scale_tomo
from tomopy.recon import algorithm as tomopy_algorithm
from tomopy.recon import wrappers

import tomopyui.gpu.recon.algorithm as tomocupy_algorithm
from tomopyui.backend.util.padding import *
from tomopyui.backend.util.registration._phase_cross_correlation_cupy import (
    phase_cross_correlation,
)


def align_joint(RunAlign):

    # ensure it only runs on 1 thread for CUDA
    os.environ["TOMOPY_PYTHON_THREADS"] = "1"

    # Initialize variables from metadata for ease of reading:
    init_tomo_shape = RunAlign.prjs.shape
    num_iter = RunAlign.num_iter
    downsample = RunAlign.downsample
    pad = RunAlign.pad
    pad_ds = RunAlign.pad_ds
    method_str = list(RunAlign.metadata.metadata["methods"].keys())[0]
    if method_str == "MLEM_CUDA":
        method_str = "EM_CUDA"
    upsample_factor = RunAlign.upsample_factor
    num_batches = RunAlign.num_batches
    center = RunAlign.center
    pre_alignment_iters = RunAlign.pre_alignment_iters
    projection_num = 50  # default to 50 now, TODO: can make an option

    # Needs scaling for skimage float operations
    RunAlign.prjs, scl = scale_tomo(RunAlign.prjs)

    # Initialization of reconstruction dataset
    tomo_shape = RunAlign.prjs.shape
    RunAlign.recon = np.mean(RunAlign.prjs) * np.empty(
        (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
    )

    # Initialize shift/convergence
    RunAlign.sx = np.zeros((tomo_shape[0]))
    RunAlign.sy = np.zeros((tomo_shape[0]))
    RunAlign.conv = np.zeros((num_iter))
    subset_x = RunAlign.subset_x
    subset_y = RunAlign.subset_y

    # Initialize projection images plot
    scale_x = bq.LinearScale(min=0, max=1)
    scale_y = bq.LinearScale(min=1, max=0)
    scales = {"x": scale_x, "y": scale_y}

    projection_fig = bq.Figure(scales=scales)
    simulated_fig = bq.Figure(scales=scales)

    scales_image = {
        "x": scale_x,
        "y": scale_y,
        "image": bq.ColorScale(
            min=float(np.min(RunAlign.prjs[projection_num])),
            max=float(np.max(RunAlign.prjs[projection_num])),
            scheme="viridis",
        ),
    }

    image_projection = ImageGL(
        image=RunAlign.prjs[projection_num],
        scales=scales_image,
    )
    image_simulated = ImageGL(
        image=np.zeros_like(RunAlign.prjs[projection_num]),
        scales=scales_image,
    )

    projection_fig.marks = (image_projection,)
    projection_fig.layout.width = "600px"
    projection_fig.layout.height = "600px"
    projection_fig.title = f"Projection Number {50}"
    simulated_fig.marks = (image_simulated,)
    simulated_fig.layout.width = "600px"
    simulated_fig.layout.height = "600px"
    simulated_fig.title = f"Re-projected Image {50}"
    with RunAlign.plot_output1:
        RunAlign.plot_output1.clear_output(wait=True)
        display(
            HBox(
                [projection_fig, simulated_fig],
                layout=Layout(
                    flex_flow="row wrap",
                    justify_content="center",
                    align_items="stretch",
                ),
            )
        )

    # Initialize Sx, Sy plot
    xs = bq.LinearScale()
    ys = bq.LinearScale()
    x = range(RunAlign.prjs.shape[0])
    y = [RunAlign.sx, RunAlign.sy]
    line = bq.Lines(
        x=x,
        y=y,
        scales={"x": xs, "y": ys},
        colors=["dodgerblue", "red"],
        stroke_width=3,
        labels=["Shift in X (px)", "Shift in Y (px)"],
        display_legend=True,
    )
    xax = bq.Axis(scale=xs, label="Projection Number", grid_lines="none")
    yax = bq.Axis(
        scale=ys,
        orientation="vertical",
        tick_format="0.1f",
        label="Shift",
        grid_lines="none",
    )
    fig_SxSy = bq.Figure(marks=[line], axes=[xax, yax], animation_duration=1000)
    fig_SxSy.layout.width = "600px"
    # Initialize convergence plot
    xs_conv = bq.LinearScale(min=0)
    ys_conv = bq.LinearScale()
    x_conv = [0]
    y_conv = [RunAlign.conv[0]]
    line_conv = bq.Lines(
        x=x_conv,
        y=y_conv,
        scales={"x": xs_conv, "y": ys_conv},
        colors=["dodgerblue"],
        stroke_width=3,
        labels=["Convergence"],
        display_legend=True,
    )
    xax_conv = bq.Axis(scale=xs_conv, label="Iteration", grid_lines="none")
    yax_conv = bq.Axis(
        scale=ys_conv,
        orientation="vertical",
        tick_format="0.1f",
        label="Convergence",
        grid_lines="none",
    )
    fig_conv = bq.Figure(
        marks=[line_conv], axes=[xax_conv, yax_conv], animation_duration=1000
    )
    fig_conv.layout.width = "600px"
    with RunAlign.plot_output2:
        RunAlign.plot_output2.clear_output()
        display(
            HBox(
                [fig_SxSy, fig_conv],
                layout=Layout(
                    flex_flow="row wrap",
                    justify_content="center",
                    align_items="stretch",
                    width="95%",
                ),
            )
        )

    # Start alignment
    for n in range(num_iter):
        if n == 0:
            recon_iterations = pre_alignment_iters
        else:
            recon_iterations = 1

        # for progress bars
        RunAlign.analysis_parent.progress_shifting.value = 0
        RunAlign.analysis_parent.progress_reprj.value = 0
        RunAlign.analysis_parent.progress_phase_cross_corr.value = 0
        _rec = RunAlign.recon
        # TODO: handle reconstruction-type parsing elsewhere
        if method_str == "SIRT_Plugin":
            RunAlign.recon = tomocupy_algorithm.recon_sirt_plugin(
                RunAlign.prjs,
                RunAlign.angles_rad,
                num_iter=recon_iterations,
                rec=_rec,
                center=center,
            )
        elif method_str == "SIRT_3D":
            RunAlign.recon = tomocupy_algorithm.recon_sirt_3D(
                RunAlign.prjs,
                RunAlign.angles_rad,
                num_iter=recon_iterations,
                rec=_rec,
                center=center,
            )
        elif method_str == "CGLS_3D":
            RunAlign.recon = tomocupy_algorithm.recon_cgls_3D_allgpu(
                RunAlign.prjs,
                RunAlign.angles_rad,
                num_iter=recon_iterations,
                rec=_rec,
                center=center,
            )
        else:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": recon_iterations,
                "extra_options": {"MinConstraint": 0},
            }
            kwargs["options"] = options
            if n == 0:
                RunAlign.recon = tomopy_algorithm.recon(
                    RunAlign.prjs,
                    RunAlign.angles_rad,
                    algorithm=wrappers.astra,
                    center=center,
                    ncore=1,
                    **kwargs,
                )
            else:
                RunAlign.recon = tomopy_algorithm.recon(
                    RunAlign.prjs,
                    RunAlign.angles_rad,
                    algorithm=wrappers.astra,
                    init_recon=_rec,
                    center=center,
                    ncore=1,
                    **kwargs,
                )

        RunAlign.recon[np.isnan(RunAlign.recon)] = 0
        RunAlign.analysis_parent.progress_total.value = n + 1
        # break up reconstruction into batches along z axis
        RunAlign.recon = np.array_split(RunAlign.recon, num_batches, axis=0)
        # may not need a copy.
        _rec = RunAlign.recon.copy()

        # initialize simulated projection cpu array
        sim = []

        # begin simulating projections using astra.
        # this could probably be made more efficient, right now I am not
        # certain if I need to be deleting every time.

        simulate_projections(
            _rec,
            sim,
            center,
            RunAlign.angles_rad,
            progress=RunAlign.analysis_parent.progress_reprj,
        )
        sim = np.concatenate(sim, axis=1)
        # only flip the simulated datasets if using normal tomopy algorithm
        # can remove if it is flipped in the algorithm
        if (
            method_str == "SIRT_Plugin"
            or method_str == "SIRT_3D"
            or method_str == "CGLS_3D"
        ):
            pass
        else:
            sim = np.flip(sim, axis=0)
            # sim = np.flip(sim, axis=2)
        # Cross correlation
        shift_cpu = []
        batch_cross_correlation(
            RunAlign.prjs,
            sim,
            shift_cpu,
            num_batches,
            upsample_factor,
            subset_correlation=RunAlign.use_subset_correlation,
            subset_x=subset_x,
            subset_y=subset_y,
            blur=True,
            pad=RunAlign.pad_ds,
            progress=RunAlign.analysis_parent.progress_phase_cross_corr,
        )
        RunAlign.shift = np.concatenate(shift_cpu, axis=1)
        # Shifting.
        (
            RunAlign.prjs,
            RunAlign.sx,
            RunAlign.sy,
            RunAlign.shift,
            err,
            RunAlign.pad_ds,
            center,
        ) = shift_prj_update_shift_cp(
            RunAlign.prjs,
            RunAlign.sx,
            RunAlign.sy,
            RunAlign.shift,
            num_batches,
            RunAlign.pad_ds,
            center,
            downsample_factor=RunAlign.ds_factor,
            progress=RunAlign.analysis_parent.progress_shifting,
        )
        RunAlign.conv[n] = np.linalg.norm(err)
        # update images
        image_projection.image = RunAlign.prjs[projection_num]
        image_simulated.image = sim[projection_num]
        # update plot lines
        line_conv.x = np.arange(0, n + 1)
        line_conv.y = RunAlign.conv[range(n + 1)]
        line.y = [RunAlign.sx * RunAlign.ds_factor, RunAlign.sy * RunAlign.ds_factor]
        RunAlign.recon = np.concatenate(RunAlign.recon, axis=0)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    # Re-normalize data
    RunAlign.prjs *= scl
    RunAlign.recon = circ_mask(RunAlign.recon, 0)
    if downsample:
        RunAlign.sx *= RunAlign.ds_factor
        RunAlign.sy *= RunAlign.ds_factor
        RunAlign.shift *= RunAlign.ds_factor

    RunAlign.pad = tuple([int(x * RunAlign.ds_factor) for x in RunAlign.pad_ds])
    return RunAlign


def simulate_projections(rec, sim, center, theta, progress=None):
    for batch in range(len(rec)):
        # for batch in tnrange(len(rec), desc="Re-projection", leave=True):
        _rec = rec[batch]
        vol_geom = astra.create_vol_geom(_rec.shape[1], _rec.shape[1], _rec.shape[0])
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
            center_shift = -(center - _rec.shape[1] / 2)
            proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
        projections_id, _sim = astra.creators.create_sino3d_gpu(
            phantom_id, proj_geom, vol_geom
        )
        _sim = _sim.swapaxes(0, 1)
        sim.append(_sim)
        astra.data3d.delete(projections_id)
        astra.data3d.delete(phantom_id)
        if progress is not None:
            progress.value += 1


def batch_cross_correlation(
    prj,
    sim,
    shift_cpu,
    num_batches,
    upsample_factor,
    blur=True,
    rin=0.5,
    rout=0.8,
    subset_correlation=False,
    subset_x=None,
    subset_y=None,
    mask_sim=True,
    pad=(0, 0),
    progress=None,
    median_filter=True,
):
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
            _prj_gpu = cp.array(
                _prj[batch][:, subset_y[0] : subset_y[1], subset_x[0] : subset_x[1]],
                dtype=cp.float32,
            )
            _sim_gpu = cp.array(
                _sim[batch][:, subset_y[0] : subset_y[1], subset_x[0] : subset_x[1]],
                dtype=cp.float32,
            )
        else:
            _prj_gpu = cp.array(_prj[batch], dtype=cp.float32)
            _sim_gpu = cp.array(_sim[batch], dtype=cp.float32)

        if median_filter:
            _prj_gpu = ndi_cp.median_filter(_prj_gpu, size=(1, 5, 5))
            _sim_gpu = ndi_cp.median_filter(_sim_gpu, size=(1, 5, 5))
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
        if progress is not None:
            progress.value += 1
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


def shift_prj_cp(
    prj, sx, sy, num_batches, pad, use_pad_cond=True, use_corr_prj_gpu=False
):
    # add checks for sx, sy having the same dimension as prj
    prj_cpu = np.array_split(prj, num_batches, axis=0)
    _sx = np.array_split(sx, num_batches, axis=0)
    _sy = np.array_split(sy, num_batches, axis=0)
    for batch in range(len(prj_cpu)):
        _prj_gpu = cp.array(prj_cpu[batch], dtype=cp.float32)
        num_theta = _prj_gpu.shape[0]
        shift_y_condition = pad[1]
        shift_x_condition = pad[0]

        for image in range(_prj_gpu.shape[0]):
            if (
                np.absolute(_sx[batch][image]) < shift_x_condition
                and np.absolute(_sy[batch][image]) < shift_y_condition
                and use_pad_cond
            ):
                shift_tuple = (_sy[batch][image], _sx[batch][image])
                _prj_gpu[image] = ndi_cp.shift(_prj_gpu[image], shift_tuple, order=5)
            elif not use_pad_cond:
                shift_tuple = (_sy[batch][image], _sx[batch][image])
                _prj_gpu[image] = ndi_cp.shift(_prj_gpu[image], shift_tuple, order=5)

        prj_cpu[batch] = cp.asnumpy(_prj_gpu)
    prj_cpu = np.concatenate(prj_cpu, axis=0)
    return prj_cpu


def shift_prj_update_shift_cp(
    prj,
    sx,
    sy,
    shift,
    num_batches,
    pad,
    center,
    downsample_factor=1,
    smart_shift=False,
    smart_pad=True,
    progress=None,
):
    # Why is the error calculated in such a strange way?
    # Will use the standard used in tomopy here, but think of different way to
    # calculate error.
    # TODO: add checks for sx, sy having the same dimension as prj
    #
    # If the shift starts to get larger than the padding in one direction,
    # shift it to the center of the sx values.
    average_sx = None
    average_sy = None
    if smart_shift:
        cond1 = sx.max() > 0.95 * pad[0]
        cond2 = sy.max() > 0.95 * pad[1]
        cond3 = np.absolute(sx.min()) > 0.95 * pad[0]
        cond4 = np.absolute(sy.min()) > 0.95 * pad[1]
        if cond1 or cond2 or cond3 or cond4:
            print("applying smart shift")
            print(f"sx max: {sx.max()}")
            print(f"sx min: {sx.min()}")
            average_sx = (sx.max() + sx.min()) / 2
            average_sy = (sy.max() + sy.min()) / 2
            sx_smart_shift = average_sx * np.ones_like(sx)
            sy_smart_shift = average_sy * np.ones_like(sy)
            sx -= sx_smart_shift
            sy -= sy_smart_shift
            print(f"sx max after shift: {sx.max()}")
            print(f"sx min after shift: {sx.min()}")
            center = center + average_sx
            if smart_pad:
                if average_sx < 1 and cond1 and cond3:
                    extra_pad = tuple([0.2 * pad[0], 0])
                    center = center + extra_pad[0]
                    pad = np.array(extra_pad) + np.array(pad)
                    prj, extra_pad = pad_projections(prj, extra_pad, 1)
                if average_sy < 1 and cond2 and cond4:
                    extra_pad = tuple([0, 0.2 * pad[1]])
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
                shift[batch][0, image] * shift[batch][0, image]
                + shift[batch][1, image] * shift[batch][1, image]
            )
            if (
                np.absolute(sx[batch][image] + shift[batch][1, image]) < pad[0]
                and np.absolute(sy[batch][image] + shift[batch][0, image]) < pad[1]
            ):
                shifted_bool[batch][image] = 1
                sx[batch][image] += shift[batch][1, image]
                sy[batch][image] += shift[batch][0, image]
                shift_tuple = (shift[batch][0, image], shift[batch][1, image])
                _prj_gpu[image] = ndi_cp.shift(_prj_gpu[image], shift_tuple, order=5)

        prj_cpu[batch] = cp.asnumpy(_prj_gpu)
        if progress is not None:
            progress.value += 1

    # concatenate the final list and return
    prj_cpu = np.concatenate(prj_cpu, axis=0)
    err = np.concatenate(err)
    shifted_bool = np.concatenate(shifted_bool)
    sx = np.concatenate(sx, axis=0)
    sy = np.concatenate(sy, axis=0)
    shift = np.concatenate(shift, axis=1)
    return prj_cpu, sx, sy, shift, err, pad, center
