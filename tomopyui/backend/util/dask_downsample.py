## This is a wrapper for skimage.transform.pyramid_reduce for large image series that
## uses dask. Pretty disorganized, but it works for my use case. Could be expanded
## eventually.

import dask
import dask.array
import dask_image
import scipy.__version__ as scipy_version
import math
import numpy as np
import scipy.ndimage as ndi

from numpy.lib import NumpyVersion

from collections.abc import Iterable


def pyramid_reduce(
    image,
    downscale=2,
    sigma=None,
    order=3,
    mode="reflect",
    cval=0.0,
    preserve_range=False,
    channel_axis=0,
):

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
    out = resize(
        smoothed, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False
    )

    return out


def _check_factor(factor):
    if factor <= 1:
        raise ValueError("scale factor must be greater than 1")


def _smooth(image, sigma, mode, cval, channel_axis):
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


def gaussian(image, sigma=1, mode="nearest", cval=0.0, truncate=4.0, channel_axis=0):
    if channel_axis is not None:
        # do not filter across channels
        if len(sigma) == image.ndim - 1:
            sigma = list(sigma)
            sigma.insert(channel_axis % image.ndim, 0)
    return dask_image.ndfilters.gaussian_filter(
        image, sigma, mode=mode, cval=cval, truncate=truncate
    )


def resize(
    image,
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
    img_bounds = [da.min(image), da.max(image)] if clip else None
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
    if NumpyVersion(scipy.__version__) >= "1.6.0":
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
        filtered = dask_image.ndinterp.spline_filter(
            padded, order, output=np.float32, mode=mode
        )
    coarsened = []
    coarsened.append(da.coarsen(np.mean, filtered, {0: 1, 1: 2, 2: 2}))
    for i in range(3):
        _ = coarsened[i]
        coarsened.append(da.coarsen(np.mean, _, {0: 1, 1: 2, 2: 2}))
    return coarsened


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
        padded = da.pad(input, npad, mode="constant", constant_values=cval)
    elif mode == "nearest":
        padded = da.pad(input, npad, mode="edge")
    else:
        # other modes have exact boundary conditions implemented so
        # no prepadding is needed
        npad = 0
        padded = input
    return padded, npad
