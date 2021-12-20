#!/usr/bin/env python

# edited from tomopy v. 1.11

import os.path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation

from tomopy.misc.corr import circ_mask
from tomopy.misc.morph import downsample
from tomopy.recon.algorithm import recon
import tomopy.util.dtype as dtype
from tomopy.util.misc import fft2, write_tiff
from tomopy.util.mproc import distribute_jobs

allowed_recon_kwargs = {
    "art": ["num_gridx", "num_gridy", "num_iter"],
    "bart": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "fbp": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
    "gridrec": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
    "mlem": ["num_gridx", "num_gridy", "num_iter"],
    "osem": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "ospml_hybrid": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "ospml_quad": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "pml_hybrid": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "pml_quad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "sirt": ["num_gridx", "num_gridy", "num_iter"],
    "tv": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "grad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "tikh": ["num_gridx", "num_gridy", "num_iter", "reg_data", "reg_par"],
}

filter_names = {
    "none",
    "shepp",
    "cosine",
    "hann",
    "hamming",
    "ramlak",
    "parzen",
    "butterworth",
}


def write_center(
    tomo,
    theta,
    cen_range=None,
    ind=None,
    num_iter=1,
    mask=False,
    ratio=1.0,
    algorithm="gridrec",
    sinogram_order=False,
    filter_name="parzen",
):

    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

    dt, dy, dx = tomo.shape
    if ind is None:
        ind = dy // 2
    if cen_range is None:
        center = np.arange(dx / 2 - 5, dx / 2 + 5, 0.5)
    else:
        center = np.arange(*cen_range)

    stack = dtype.empty_shared_array((len(center), dt, dx))

    for m in range(center.size):
        if sinogram_order:
            stack[m] = tomo[ind]
        else:
            stack[m] = tomo[:, ind, :]

    # Reconstruct the same slice with a range of centers.
    if algorithm == "gridrec" or algorithm == "fbp":
        rec = recon(
            stack,
            theta,
            center=center,
            sinogram_order=True,
            algorithm=algorithm,
            filter_name=filter_name,
            nchunk=1,
        )
    else:
        rec = recon(
            stack,
            theta,
            center=center,
            sinogram_order=True,
            algorithm=algorithm,
            num_iter=num_iter,
            ncore=None,
            nchunk=1,
        )

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    return rec, center
