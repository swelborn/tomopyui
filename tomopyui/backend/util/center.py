#!/usr/bin/env python

# edited from tomopy v. 1.11

import os.path

import numpy as np

from tomopy.misc.corr import circ_mask
from tomopy.recon.algorithm import recon
import tomopy.util.dtype as dtype
# includes astra_cuda_recon_algorithm_kwargs, tomopy_recon_algorithm_kwargs,
# and tomopy_filter_names, extend_description_style
from tomopyui._sharedvars import *

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
