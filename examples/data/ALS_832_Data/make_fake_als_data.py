#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import h5py
import os
import dxchange


# In[59]:


filename = "./fake_als_data.h5"

numslices = 100
numrays = 120
numangles = 80

tomo = np.random.rand(numangles, numslices, numrays)
flat = np.ones((5, numslices, numrays))  # first dimension can be changed to anything
dark = np.zeros((5, numslices, numrays))  # first dimension can be changed to anything
angles = np.linspace(0, np.pi, num=numangles)


# In[60]:


pxsize = 1  # in mm
camera_distance = 100 * np.ones(len(tomo) + len(flat) + len(dark))  # in mm
energy = 10000  # in eV


# In[61]:


if os.path.exists(filename):
    os.remove(filename)
with h5py.File(filename, "a") as f:
    det = f.create_group("measurement/instrument/detector")
    det.create_dataset("dimension_y", data=np.asarray(numslices)[np.newaxis])
    det.create_dataset("dimension_x", data=np.asarray(numrays)[np.newaxis])
    det.create_dataset("pixel_size", data=np.asarray(pxsize)[np.newaxis])
    rot = f.create_group("process/acquisition/rotation")
    rot.create_dataset("num_angles", data=np.asarray(len(angles))[np.newaxis])
    rot.create_dataset(
        "range",
        data=np.asarray((180 / np.pi) * np.abs(angles[-1] - angles[0]))[np.newaxis],
    )
    f.create_dataset(
        "measurement/instrument/camera_motor_stack/setup/camera_distance",
        data=camera_distance,
    )
    f.create_dataset(
        "measurement/instrument/monochromator/energy",
        data=np.asarray(energy)[np.newaxis],
    )
    exch = f.create_group("exchange")
    exch.create_dataset("data", data=tomo)
    exch.create_dataset("data_white", data=flat)
    exch.create_dataset("data_dark", data=dark)
    exch.create_dataset("theta", data=(180 / np.pi) * angles)


# In[62]:


numslices = int(
    dxchange.read_hdf5(filename, "/measurement/instrument/detector/dimension_y")[0]
)
numrays = int(
    dxchange.read_hdf5(filename, "/measurement/instrument/detector/dimension_x")[0]
)
pxsize = (
    dxchange.read_hdf5(filename, "/measurement/instrument/detector/pixel_size")[0]
    / 10.0
)  # /10 to convert units from mm to cm
numangles = int(
    dxchange.read_hdf5(filename, "/process/acquisition/rotation/num_angles")[0]
)
propagation_dist = dxchange.read_hdf5(
    filename, "/measurement/instrument/camera_motor_stack/setup/camera_distance"
)[1]
kev = (
    dxchange.read_hdf5(filename, "/measurement/instrument/monochromator/energy")[0]
    / 1000
)
angularrange = dxchange.read_hdf5(filename, "/process/acquisition/rotation/range")[0]


# In[63]:


tomo, flat, dark, angles = dxchange.exchange.read_aps_tomoscan_hdf5(filename)
print(tomo.shape, tomo.dtype, tomo.min(), tomo.max())
print(flat.shape, flat.dtype, flat.min(), flat.max())
print(dark.shape, dark.dtype, dark.min(), dark.max())
print(angles.shape, angles.dtype, angles.min(), angles.max())


# In[ ]:


# In[ ]:
