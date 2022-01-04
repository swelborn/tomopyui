import numpy as np

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
