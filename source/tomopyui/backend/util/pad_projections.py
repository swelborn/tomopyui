import numpy as np

def pad_projections(prj, pad, downsample_factor):
    pad_ds = tuple([int(downsample_factor*x) for x in pad])
    npad_ds = ((0, 0), (pad_ds[1], pad_ds[1]), (pad_ds[0], pad_ds[0]))
    prj = np.pad(
        prj, npad_ds, mode="constant", constant_values=0
    )
    return prj, pad_ds