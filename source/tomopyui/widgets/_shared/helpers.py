import os
import glob
import numpy as np
import json
import functools
import tifffile as tf
from ipywidgets import *

def get_img_shape(fpath, fname, ftype, metadata, folder_import=False):
    '''

    '''

    os.chdir(fpath)
    if ftype=="tiff":
        tiff_count_in_folder = len(glob.glob1(fpath, "*.tif"))
        # TODO: make this so it just reads the number of tiffs in the folder 
        # and the first tiff to avoid loading data into memory.
        if folder_import:
            _tomo = td.TomoData(metadata=metadata)
            size = _tomo.prj_imgs.shape
            sizeZ = size[0]
            sizeY = size[1]
            sizeX = size[2]
        else:
            with tf.TiffFile(fname) as tif:
                # if you select a file instead of a file path, it will try to
                # bring in the full folder
                # this may cause issues if someone is trying to bring in a 
                # file in a folder with a lot of tiffs. can just make a note
                # to do the analysis in a 'fresh' folder 
                if tiff_count_in_folder > 50:
                    sizeX = tif.pages[0].tags["ImageWidth"].value
                    sizeY = tif.pages[0].tags["ImageLength"].value
                    sizeZ = tiff_count_in_folder # can maybe use this later
                else:
                    imagesize = tif.pages[0].tags["ImageDescription"]
                    size = json.loads(imagesize.value)["shape"]
                    sizeZ = size[0]
                    sizeY = size[1]
                    sizeX = size[2]

    elif ftype == "npy":
        size = np.load(fname, mmap_mode="r").shape
        sizeY = size[1]
        sizeX = size[2]

    return (sizeZ, sizeY, sizeX)

class MetaCheckbox:
    def __init__(self, description, dictionary, obj, disabled=False, value=False):
        
        self.checkbox = Checkbox(
                        description=description,
                        value=value,
                        disabled=disabled
                        )

        def create_opt_dict_on_check(change):
            dictionary[description] = change.new
            obj.set_metadata() # obj needs a set_metadata function
            
        self.checkbox.observe(create_opt_dict_on_check, names="value")

def create_checkbox(description, disabled=False, value=False):
    checkbox = Checkbox(description=description, disabled=disabled, value=value)
    return checkbox

def create_checkboxes_from_opt_list(opt_list, dictionary, obj):
    checkboxes = [MetaCheckbox(opt, dictionary, obj) for opt in opt_list]
    return [a.checkbox for a in checkboxes] # return list of checkboxes 

