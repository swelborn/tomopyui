import numpy as np
from dask.array import core as da_core
from tomopyui.backend.helpers import (
    np_hist,
    dask_hist,
)  # Adjust import based on your project structure


def test_histogram_functions():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate a random numpy array
    np_data = np.random.rand(10)
    # Convert numpy array to dask array
    da_data = da_core.from_array(np_data, chunks=(1,))

    # Get results from np_hist
    np_hist_result, np_range, np_bins, da_centers, np_percentile = np_hist(np_data)
    np.testing.assert_array_equal(
        np_data,
        da_data.compute(),
        err_msg="Input arrays to np_hist and dask_hist are not identical",
    )

    # Get results from dask_hist
    da_hist_result, da_range, da_bins, da_centers, da_percentile = dask_hist(da_data)
    # Ensure computation for dask results
    da_hist_result = (da_hist_result[0].compute(), da_hist_result[1].compute())
    da_range = (da_range[0].compute(), da_range[1].compute())
    da_percentile = da_percentile.compute()

    # Check output types for np_hist
    assert isinstance(np_hist_result, tuple), "Expected np_hist_result to be a tuple"
    assert isinstance(
        np_hist_result[0], np.ndarray
    ), "Expected np_hist_result[0] to be a np.ndarray"
    assert isinstance(
        np_hist_result[1], np.ndarray
    ), "Expected np_hist_result[1] to be a np.ndarray"
    assert isinstance(np_range, tuple), "Expected np_range to be a tuple"
    assert isinstance(np_bins, int), "Expected np_bins to be an int"
    assert isinstance(
        np_percentile, np.ndarray
    ), "Expected np_percentile to be a np.ndarray"
    assert len(np_percentile) == 2, "Expected np_percentile to contain two elements"

    # Check output types for dask_hist
    assert isinstance(da_hist_result, tuple), "Expected da_hist_result to be a tuple"
    assert isinstance(
        da_hist_result[0], np.ndarray
    ), "Expected da_hist_result[0] to be a np.ndarray after computation"
    assert isinstance(
        da_hist_result[1], np.ndarray
    ), "Expected da_hist_result[1] to be a np.ndarray after computation"
    assert isinstance(
        da_range, tuple
    ), "Expected da_range to be a tuple after computation"
    assert isinstance(da_bins, int), "Expected da_bins to be an int"
    assert isinstance(
        da_percentile, np.ndarray
    ), "Expected da_percentile to be a np.ndarray after computation"
    assert (
        len(da_percentile) == 2
    ), "Expected da_percentile to contain two elements after computation"

    # Check for consistency between np_hist and dask_hist results
    np.testing.assert_array_almost_equal(
        np_hist_result[0],
        da_hist_result[0],
        decimal=5,
        err_msg="Histogram counts do not match",
    )
    np.testing.assert_array_almost_equal(
        np_hist_result[1],
        da_hist_result[1],
        decimal=5,
        err_msg="Histogram bin edges do not match",
    )
    print(np_percentile, da_percentile)
    np.testing.assert_array_almost_equal(
        np_percentile, da_percentile, decimal=1, err_msg="Percentiles do not match"
    )
