#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with abstraction from core tomopy functions. Includes classes TomoData 
and Recon, which store useful information about the projections and
reconstructions. Can use these classes for plotting. Written for use in 
Jupyter notebook found in doc/demo (TODO: add Jupyter notebook here)

"""

from __future__ import print_function

import logging
import numexpr as ne
import dxchange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import smtplib
import time
import os
import tifffile as tf
import glob

from tomopy.sim.project import angles as angle_maker
from matplotlib import animation, rc, colors
from matplotlib.widgets import Slider


# ----------------------------- Class TomoData -------------------------#


class TomoData:
    def __init__(
        self,
        prj_imgs=None,
        numX=None,
        numY=None,
        num_theta=None,
        theta=None,
        verbose_import=False,
        metadata=None,
        fname=None,
        fpath=None,
        angle_start=None,
        angle_end=None
        # correctionOptions=dict(),
    ):
        self.metadata = metadata
        self.prj_imgs = prj_imgs
        self.numX = numX
        self.numY = numY
        self.num_theta = num_theta
        self.theta = theta
        self.verbose_import = verbose_import
        self.fpath = fpath
        self.fname = fname

        if self.verbose_import == True:
            logging.getLogger("dxchange").setLevel(logging.INFO)
        else:
            logging.getLogger("dxchange").setLevel(logging.WARNING)

        if self.metadata is not None and self.prj_imgs is None:
            self.fpath = self.metadata["fpath"]
            self.fname = self.metadata["fname"]
            self.metadata["imgtype"] = ""
            self.filetype_parser()
            if self.metadata["imgtype"] == "tiff":
                self = self.import_tiff()
            if self.metadata["imgtype"] == "tiff folder":
                self = self.import_tiff_folder()
            if self.metadata["imgtype"] == "npy":
                self = self.import_npy()

        if self.prj_imgs is not None:
            self.num_theta, self.numY, self.numX = self.prj_imgs.shape
        # can probably fix this later to rely on user input. Right now user
        # input is only for storing metadata, maybe better that way.
        if self.theta is None and self.num_theta is not None:
            self.theta = angle_maker(
                self.num_theta,
                ang1=self.metadata["angle_start"],
                ang2=self.metadata["angle_end"],
            )

        if self.prj_imgs is None:
            logging.warning("This did not import.")

    # --------------------------Import Functions--------------------------#

    def filetype_parser(self):
        if self.fname == "":
            self.metadata["imgtype"] = "tiff folder"
        if self.fname.__contains__(".tif"):
            # if there is a file name, checks to see if there are many more
            # tiffs in the folder. If there are, will upload all of them.
            tiff_count_in_folder = len(glob.glob1(self.fpath, "*.tif"))
            if tiff_count_in_folder > 50:
                self.metadata["imgtype"] = "tiff folder"
            else:
                self.metadata["imgtype"] = "tiff"
        if self.fname.__contains__(".npy"):
            self.metadata["imgtype"] = "npy"

        return self

    def import_tiff(self):
        """
        Import tiff and create TomoData object based on option_dict.

        Returns
        -------
        self : TomoData
        """
        # navigates to path selected. User may pick a file instead of a folder.
        os.chdir(self.metadata["fpath"])
        self.prj_imgs = dxchange.reader.read_tiff(self.metadata["fname"]).astype(
            np.float32
        )
        if self.prj_imgs.ndim == 2:
            self.prj_imgs = self.prj_imgs[np.newaxis, :, :]
        # this will rotate it 90 degrees. Can update to rotate it multiple
        # times.
        if "rotate" in self.metadata:
            if self.metadata["rotate"]:
                self.prj_imgs = np.swapaxes(self.prj_imgs, 1, 2)
                self.prj_imgs = np.flip(self.prj_imgs, 2)
        return self

    def import_tiff_folder(self, num_theta=None):
        """
        Import tiffs in a folder.

        Parameters
        ----------
        num_theta: int, required
            Total number of projection images taken.
        Returns
        -------
        self : TomoData
        """
        # navigates to path selected. User may pick a file instead of a folder.
        # This should not matter, it ignores that.

        os.chdir(self.metadata["fpath"])
        # Using tiffsequence instead of dxchange. dxchange.read_tiff_stack
        # does not do a good job finding files if they do not have a number
        # at the end.

        image_sequence = tf.TiffSequence()
        self.num_theta = len(image_sequence.files)
        self.prj_imgs = image_sequence.asarray().astype(np.float32)
        image_sequence.close()
        # rotate dataset 90 deg if wanted
        if "rotate" in self.metadata:
            if self.metadata["rotate"]:
                self.prj_imgs = np.swapaxes(self.prj_imgs, 1, 2)
                self.prj_imgs = np.flip(self.prj_imgs, 2)

        return self

    def import_npy(self):
        """
        Import tiff and create TomoData object based on option_dict.

        Returns
        -------
        self : TomoData
        """
        # navigates to path selected. User may pick a file instead of a folder.
        os.chdir(self.metadata["fpath"])
        self.prj_imgs = np.load(self.metadata["fname"]).astype(np.float32)
        if self.prj_imgs.ndim == 2:
            self.prj_imgs = self.prj_imgs[np.newaxis, :, :]
        # this will rotate it 90 degrees. Can update to rotate it multiple
        # times.
        if "rotate" in self.metadata:
            if self.metadata["opts"]:
                self.prj_imgs = np.swapaxes(self.prj_imgs, 1, 2)
                self.prj_imgs = np.flip(self.prj_imgs, 2)
        return self

    def writeTiff(self, fname):
        """
        Writes prj_imgs attribute to file.

        Parameters
        ----------
        fname : str, relative or absolute filepath.

        """
        dxchange.write_tiff(self.prj_imgs, fname=fname)

    # --------------------------Correction Functions--------------------------#

    def removeStripes(self, options):
        """
        Remove stripes from sinograms so that you end up with less ring
        artifacts in reconstruction.

        eg

        .. highlight:: python
        .. code-block:: python

            tomoCorr = tomo.removeStripes(
                options={
                    "remove_all_stripe": {
                        "snr": 3,
                        "la_size": 61,
                        "sm_size": 21,
                        "dim": 1,
                        "ncore": None,
                        "nchunk": None,
                    },
                    "remove_large_stripe": {
                        "snr": 3,
                        "size": 51,
                        "drop_ratio": 0.1,
                        "norm": True,
                        "ncore": None,
                        "nchunk": None,
                    },
                }
            )

        Parameters
        ----------
        options : nested dict

            The formatting here is important - the keys in the 0th level of
            the dictionary (i.e. 'remove_all_stripe') will call
            the tomopy.prep.stripe function with the same name. Its
            corresponding value are the options input into that function.
            The order of operations will proceed with the first dictionary
            key given, then the second, and so on...

        Returns
        -------
        self : tomoData
        """
        for key in options:
            if key == "remove_all_stripe":
                print("Performing ALL stripe removal.")
                self.prj_imgs = tomopy.prep.stripe.remove_all_stripe(
                    self.prj_imgs, **options[key]
                )
            if key == "remove_large_stripe":
                print("Performing LARGE stripe removal.")
                self.prj_imgs = tomopy.prep.stripe.remove_large_stripe(
                    self.prj_imgs, **options[key]
                )
        self.correctionOptions = options
        return self


############################# TomoDataCombined #############################


def normalize(tomo, flat, dark, rmZerosAndNans=True):
    """
    Normalizes the data with typical options for normalization. TODO: Needs
    more options.
    TODO: add option to delete the previous objects from memory.

    Parameters
    ----------
    rmZerosAndNans : bool
        Remove the zeros and nans from normalized data.

    Returns
    -------
    tomoNorm : TomoData
        Normalized data in TomoData object
    tomoNormMLog : TomoData
        Normalized + -log() data in TomoData object
    """
    tomoNormprj_imgs = tomopy.normalize(tomo.prj_imgs, flat.prj_imgs, dark.prj_imgs)
    tomoNorm = TomoData(prj_imgs=tomoNormprj_imgs, raw="No")
    tomoNormMLogprj_imgs = tomopy.minus_log(tomoNormprj_imgs)
    tomoNormMLog = TomoData(prj_imgs=tomoNormMLogprj_imgs, raw="No")
    if rmZerosAndNans == True:
        tomoNormMLog.prj_imgs = tomopy.misc.corr.remove_nan(
            tomoNormMLog.prj_imgs, val=0.0
        )
        tomoNormMLog.prj_imgs[tomoNormMLog.prj_imgs == np.inf] = 0
    return tomoNorm, tomoNormMLog


################################### Misc. Functions ###########################


def textme(phoneNumber, carrierEmail, gmail_user, gmail_password):
    """
    From https://stackabuse.com/how-to-send-emails-with-gmail-using-python/.

    Sends a text message to you when called.

    Parameters
    ----------
    phoneNumber : str
    carrierEmail : this is your carrier email. TODO, find list of these, and
        allow input of just the carrier. Idea from TXMWizard software.
    gmail_user : str, gmail username
    gmail_password : str, gmail password
    """

    toaddr = str(phoneNumber + "@" + carrierEmail)
    fromaddr = gmail_user
    message_subject = "Job done."
    message_text = "Finished the job."
    message = (
        "From: %s\r\n" % fromaddr
        + "To: %s\r\n" % toaddr
        + "Subject: %s\r\n" % message_subject
        + "\r\n"
        + message_text
    )
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(fromaddr, toaddr, message)
        server.close()
    except:
        print("Something went wrong...")
