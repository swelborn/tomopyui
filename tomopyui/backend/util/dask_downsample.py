## This is a wrapper for skimage.transform.pyramid_reduce for large image series that
## uses dask. Pretty disorganized, but it works for my use case. Could be expanded
## eventually.

# TODO: currently, spline filter does not work properly - it is also splining the z axis
# USE multiple 1d splines


import math
from tomopyui.backend.hdf_manager import (
    hdf_key_ds,
)
from dask.array import core as da_core
from dask.array import routines as da_rout
from dask.array import creation as da_create
from tomopyui.backend.helpers import dask_hist
from tomopyui.backend.helpers import DaskHistOutput
from dask_image.ndfilters import gaussian_filter
from dask_image.ndinterp import spline_filter1d
import numpy as np
from typing import Optional
from tomopyui.backend.hdf_manager import HDFManager
from numpy.lib import NumpyVersion
from scipy import __version__ as scipy_version
from collections.abc import Iterable


def _prepare_image(image: da_core.Array, channel_axis: int) -> da_core.Array:
    """
    Prepares the image for processing, including rechunking.

    Parameters:
    - image: dask.array.Array - The input image.
    - channel_axis: int - The axis that represents color channels.

    Returns:
    - Prepared dask.array.Array.
    """
    if not isinstance(image, da_core.Array):
        image = da_core.from_array(image, chunks="auto")
    else:
        image = image.rechunk(chunks="auto")
    return image


def pyramid_reduce_gaussian(
    image: Optional[da_core.Array] = None,
    hdf_manager: Optional["HDFManager"] = None,
    downscale: int = 2,
    sigma: Optional[float] = None,
    order: int = 3,
    mode: str = "reflect",
    cval: float = 0.0,
    preserve_range: bool = False,
    channel_axis: int = 0,
    pyramid_levels: int = 3,
    compute: bool = False,
) -> Optional[list[da_core.Array]]:
    """
    Wrapper for skimage.transform.pyramid_reduce_gaussian.

    Parameters
    ----------
    image: dask.array
        Time series images that you want to reduce into pyramid form.
    downscale: int
        Factor by which you would like to downscale the image (along "X" and "Y" pixels)
    sigma
        Gaussian standard deviation. Will apply equally on x and y, but not on the channel
        axis (which defaults to 0).
    order: int
        Found in pyramid_reduce order description.
    mode: str
        Check pyramid_reduce for description.
    cval
        Defines constant value added to borders.
    pyramid_levels: int
        Number of levels to downscale by 2.
    """

    # Initialization
    coarsened_images = []
    histograms = []

    # Delete stuff if it is already there
    if hdf_manager is None and image is None:
        return

    data = None
    if hdf_manager:
        if hdf_manager.mode != "r+":
            print("HDF5 file is not open for writing.")
            return
        ds_data = hdf_manager.get_ds_data(pyramid_level=0, lazy=True)
        if ds_data is not None:
            hdf_manager.delete_ds_data()
        data = hdf_manager.get_normalized_data(lazy=True)

    image = data if isinstance(data, da_core.Array) else image

    if image is None:
        print("No image to downsample...")
        return None

    image = _prepare_image(image, channel_axis)
    # image.compute()

    # Process pyramid levels
    for level in range(pyramid_levels):
        image, hist = _process_pyramid_level(
            image,
            downscale,
            sigma,
            order,
            mode,
            cval,
            preserve_range,
            channel_axis,
            level,
            hdf_manager,
        )
        if image is None:
            break
        coarsened_images.append(image)
        histograms.append(hist)

    # Optionally compute results
    if compute:
        coarsened_images = [img.compute() for img in coarsened_images]

    return coarsened_images


def _process_pyramid_level(
    image: da_core.Array,
    downscale: int,
    sigma: Optional[float],
    order: int,
    mode: str,
    cval: float,
    preserve_range: bool,
    channel_axis: int,
    level: int,
    hdf_manager: Optional[HDFManager],
) -> tuple[Optional[da_core.Array], Optional[DaskHistOutput]]:

    pad_on_levels = _check_divisible(image, 2)
    if pad_on_levels is not None:
        image = da_create.pad(image, pad_on_levels)
    filtered = pyramid_reduce(
        image,
        downscale=downscale,
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        channel_axis=channel_axis,
    )

    if filtered is None:
        return None, None

    coarsened = da_rout.coarsen(np.mean, filtered, {0: 1, 1: 2, 2: 2}).astype(
        np.float32
    )
    hist_out = dask_hist(coarsened)

    subgrp = hdf_key_ds + str(level) + "/"
    if hdf_manager:
        hdf_manager.save_hist_and_data(hist_out, coarsened, subgrp)

    return coarsened, hist_out


def pyramid_reduce(
    image,
    downscale: int = 2,
    sigma: Optional[float] = None,
    order: int = 3,
    mode: str = "reflect",
    cval: float = 0.0,
    preserve_range: bool = False,
    channel_axis: int = 0,
) -> Optional[da_core.Array]:
    """
    Wrapper for skimage.transform.pyramid_reduce.

    Parameters
    ----------
    image: dask.array
            Time series images that you want to reduce into pyramid form.
    downscale: int
            Factor by which you would like to downscale the image (along "X" and "Y" pixels)
    sigma
            Gaussian standard deviation. Will apply equally on x and y, but not on the channel
            axis (which defaults to 0).
    order: int
            Found in pyramid_reduce order description.
    mode: str
            Padding mode? Check pyramid_reduce for description.
    cval
            Defines constant value added to
    """
    _check_factor(downscale)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple(
            math.ceil(d / float(downscale)) if ax != channel_axis else d
            for ax, d in enumerate(image.shape)
        )
    else:
        out_shape = tuple(math.ceil(d / float(downscale)) for d in image.shape)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    smoothed = _smooth(image, sigma, mode, cval, channel_axis)
    # TODO: change names. Resize only spline interpolates the data right now.
    try:
        filtered = resize(
            smoothed, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False
        )
    except ValueError as e:
        return
    else:
        return filtered


def _check_divisible(arr: da_core.Array, factor: int):
    shape = arr.shape
    shape_mod = [dim % 2 for dim in shape]
    if any(x != 0 for x in shape_mod):
        pad = [0 if mod == 0 else 1 for mod in shape_mod]
        pad[0] = 0
        pad = [(0, p) for p in pad]
        return pad
    else:
        return None


def _check_factor(factor):
    if factor <= 1:
        raise ValueError("scale factor must be greater than 1")


def _smooth(
    image: da_core.Array,
    sigma: float = 1.0,
    mode: str = "nearest",
    cval: float = 0.0,
    channel_axis: Optional[int] = 0,
):
    """Return image with each channel smoothed by the Gaussian filter."""
    # apply Gaussian filter to all channels independently
    if channel_axis is not None:
        # can rely on gaussian to insert a 0 entry at channel_axis
        channel_axis = channel_axis % image.ndim
        sigma = (sigma,) * (image.ndim - 1)
    else:
        channel_axis = None
    smoothed = gaussian(image, sigma, mode=mode, cval=cval, channel_axis=channel_axis)
    return smoothed


def gaussian(
    image: da_core.Array,
    sigma: tuple[float, ...] = 1,
    mode: str = "nearest",
    cval: float = 0.0,
    truncate: float = 4.0,
    channel_axis: Optional[int] = 0,
):
    if channel_axis is not None:
        # do not filter across channels
        if len(sigma) == image.ndim - 1:
            sigma_list = list(sigma)
            sigma_list.insert(channel_axis % image.ndim, 0)
    return gaussian_filter(image, sigma_list, mode=mode, cval=cval, truncate=truncate)


def resize(
    image: da_core.Array,
    output_shape,
    order=3,
    mode="reflect",
    cval=0.0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
):
    image, output_shape = _preprocess_resize_output_shape(image, output_shape)
    input_shape = image.shape
    input_type = image.dtype
    if input_type == np.float16:
        image = image.astype(np.float32)

    if anti_aliasing is None:
        anti_aliasing = not input_type == bool and any(
            x < y for x, y in zip(output_shape, input_shape)
        )

    if input_type == bool and anti_aliasing:
        raise ValueError("anti_aliasing must be False for boolean images")
    factors = np.divide(input_shape, output_shape)
    # Save input value range for clip
    # img_bounds = [da_red.min(image), da.max(image)] if clip else None
    # Translate modes used by np.pad to those used by scipy.ndimage
    ndi_mode = _to_ndimage_mode(mode)
    if NumpyVersion(scipy_version) >= "1.6.0":
        # The grid_mode kwarg was introduced in SciPy 1.6.0
        zoom_factors = [1 / f for f in factors]
        out = zoom(image, zoom_factors, mode=ndi_mode, cval=cval, grid_mode=True)
    return out


def _preprocess_resize_output_shape(image, output_shape):
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1,) * (output_ndim - image.ndim)
    elif output_ndim == image.ndim - 1:
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        raise ValueError(
            "output_shape length cannot be smaller than the "
            "image number of dimensions"
        )

    return image, output_shape


def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(
        constant="constant",
        edge="nearest",
        symmetric="reflect",
        reflect="mirror",
        wrap="wrap",
    )
    if mode not in mode_translation_dict:
        raise ValueError(
            (
                f"Unknown mode: '{mode}', or cannot translate mode. The "
                f"mode should be one of 'constant', 'edge', 'symmetric', "
                f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
                f"more info."
            )
        )
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {"constant": "grid-constant", "wrap": "grid-wrap"}
    if NumpyVersion(scipy_version) >= "1.6.0":
        mode = grid_modes.get(mode, mode)
    return mode


def zoom(
    input, zoom, order=3, mode="constant", cval=0.0, prefilter=True, grid_mode=False
):
    if order < 0 or order > 5:
        raise RuntimeError("spline order not supported")
    if input.ndim < 1:
        raise RuntimeError("input and output rank must be > 0")
    zoom = _normalize_sequence(zoom, input.ndim)
    output_shape = tuple([int(round(ii * jj)) for ii, jj in zip(input.shape, zoom)])
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter1d(padded, order, axis=1, mode=mode)
        filtered = spline_filter1d(padded, order, axis=2, mode=mode)
    return filtered


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized


def _prepad_for_spline_filter(input, mode, cval):
    if mode in ["nearest", "grid-constant"]:
        npad = 12
    if mode == "grid-constant":
        padded = da_create.pad(input, npad, mode="constant", constant_values=cval)
    elif mode == "nearest":
        padded = da_create.pad(input, npad, mode="edge")
    else:
        # other modes have exact boundary conditions implemented so
        # no prepadding is needed
        npad = 0
        padded = input
    return padded, npad
