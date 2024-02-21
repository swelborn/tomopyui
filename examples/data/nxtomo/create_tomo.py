import os
import pathlib

import dxchange
import numpy
from nxtomo import NXtomo
from nxtomo.nxobject.nxdetector import ImageKey
from tomoscan.esrf.scan.nxtomoscan import NXtomoScan

THIS_PATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = pathlib.Path(THIS_PATH / "tomo_00077.h5")  # from tomobank

# I took most of this from the tomoscan tutorial, but I had to modify it.
# I am not sure how you are storing your "projections" data. Basically, we want
# projection images along x and y in a [Z, Y, X] array, like is output by dxchange.read_aps_32id.
# You can modify your data import method in tomopyui.backend.io to get it into the right format.
proj, flat, dark, theta = dxchange.read_aps_32id(fname=DATA_PATH, proj=(0, 477))

binning = 8
proj_binned = proj[:, ::binning, ::binning]
dark_binned = dark[:, ::binning, ::binning]
flat_binned = flat[:, ::binning, ::binning]
assert proj_binned.shape[2] == dark_binned.shape[2] == flat_binned.shape[2]
assert proj_binned.shape[1] == dark_binned.shape[1] == flat_binned.shape[1]
proj_rotation_angles = theta * 180 / numpy.pi
assert len(proj_rotation_angles) == len(proj_binned)

my_nxtomo = NXtomo()

# create the array
data = numpy.concatenate(
    [
        dark_binned,
        flat_binned,
        proj_binned,
    ]
)
assert data.ndim == 3
print(data.shape)
# then register the data to the detector
my_nxtomo.instrument.detector.data = data

image_key_control = numpy.concatenate(
    [
        [ImageKey.DARK_FIELD] * len(dark_binned),
        [ImageKey.FLAT_FIELD] * len(flat_binned),
        [ImageKey.PROJECTION] * len(proj_binned),
    ]
)

# insure with have the same number of frames and image key
assert len(image_key_control) == len(data)
# print position of flats in the sequence
print("flats indexes are", numpy.where(image_key_control == ImageKey.FLAT_FIELD))
# then register the image keys to the detector
my_nxtomo.instrument.detector.image_key_control = image_key_control

rotation_angle = numpy.concatenate(
    [
        [0 for x in range(len(dark_binned))],
        [0 for x in range(len(flat_binned))],
        proj_rotation_angles,
    ]
)
assert len(rotation_angle) == len(data)
# register rotation angle to the sample
my_nxtomo.sample.rotation_angle = rotation_angle

my_nxtomo.instrument.detector.field_of_view = "Full"

my_nxtomo.instrument.detector.x_pixel_size = (
    my_nxtomo.instrument.detector.y_pixel_size
) = 1e-7  # pixel size must be provided in SI: meter
my_nxtomo.instrument.detector.x_pixel_size = (
    my_nxtomo.instrument.detector.y_pixel_size
) = 0.1
my_nxtomo.instrument.detector.x_pixel_size.unit = (
    my_nxtomo.instrument.detector.y_pixel_size.unit
) = "micrometer"

nx_tomo_file_path = pathlib.Path(THIS_PATH / "tomo_00077.nx")
my_nxtomo.save(file_path=str(nx_tomo_file_path), data_path="entry", overwrite=True)

has_tomoscan = False
try:
    import tomoscan
except ImportError:
    has_tomoscan = False
    from tomoscan.esrf import NXtomoScan
    from tomoscan.validator import ReconstructionValidator

    has_tomoscan = True

if has_tomoscan:
    scan = NXtomoScan(nx_tomo_file_path, entry="entry")
    validator = ReconstructionValidator(
        scan, check_phase_retrieval=False, check_values=True
    )
    assert validator.is_valid()
