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
from tomopy.sim.project import angles as angle_maker
import smtplib
import time
import os
import tifffile as tf
import glob

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
        filename=None,
        theta=None,
        cbarRange=[0, 1],
        verbose_import=False,
        metadata=None,
        # correctionOptions=dict(),
    ):
        self.metadata = metadata
        self.prj_imgs = prj_imgs
        self.numX = numX
        self.numY = numY
        self.num_theta = num_theta
        self.theta = theta
        self.verbose_import = verbose_import
        self.filename = filename
        self.cbarRange = cbarRange

        if self.verbose_import == True:
            logging.getLogger("dxchange").setLevel(logging.INFO)
        else:
            logging.getLogger("dxchange").setLevel(logging.WARNING)

        if self.metadata is not None and self.prj_imgs is None:
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
                self.num_theta, self.metadata["start_angle"], self.metadata["end_angle"]
            )

        if self.prj_imgs is None:
            logging.warning("This did not import.")

    # --------------------------Import Functions--------------------------#

    def filetype_parser(self):
        fpath = self.metadata["fpath"]
        fname = self.metadata["fname"]
        if fname == "":
            self.metadata["imgtype"] = "tiff folder"
        if fname.__contains__(".tif"):
            # if there is a file name, checks to see if there are many more
            # tiffs in the folder. If there are, will upload all of them.
            tiff_count_in_folder = len(glob.glob1(fpath, "*.tif"))
            if tiff_count_in_folder > 50:
                self.metadata["imgtype"] = "tiff folder"
            else:
                self.metadata["imgtype"] = "tiff"
        if fname.__contains__(".npy"):
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
        if "opts" in self.metadata:
            if "rotate" in self.metadata["opts"]:
                if self.metadata["opts"]["rotate"]:
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
        if "opts" in self.metadata:
            if "rotate" in self.metadata["opts"]:
                if self.metadata["opts"]["rotate"]:
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
        if "opts" in self.metadata:
            if "rotate" in self.metadata["opts"]:
                if self.metadata["opts"]["rotate"]:
                    self.prj_imgs = np.swapaxes(self.prj_imgs, 1, 2)
                    self.prj_imgs = np.flip(self.prj_imgs, 2)
        return self

    # --------------------------Plotting Functions----------------------#

    def plotProjectionImage(
        self, projectionNo=0, figSize=(8, 4), cmap="viridis", cmapRange=None
    ):
        """
        Plot a specific projection image.
        This has controls so that you can plot the image and set the correct
        color map range. The colormap range set here can be used to plot a
        movie using plotProjectionMovie.

        Sliders idea taken from: https://stackoverflow.com/questions/65040676/
        matplotlib-sliders-rescale-colorbar-when-used-to-change-clim

        Parameters
        ----------
        projectionNo : int
            Must be from 0 to the total number of projection images you took.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str
            Colormap of choice. You can choose from the ones on
            matplotlib.colors
        cmapRange : list with 2 entries, [0,1]
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        if len(self.prj_imgs.shape) == 3:
            imgData = self.prj_imgs[projectionNo, :, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        if cmapRange == None:
            c_min = np.min(imgData)  # the min and max range of the sliders
            c_max = np.max(imgData)
        else:
            c_min = cmapRange[0]
            c_max = cmapRange[1]

        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # define sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)

        plt.show()

    def plotSinogram(
        self, sinogramNo=0, figSize=(8, 4), cmap="viridis", cmapRange=None
    ):
        """
        Plot a sinogram given the sinogram number. The slice for one particular
        value of y pixel. This has controls so that you can plot the image and
        set the correct color map range. The colormap range set here can be
        used to plot a movie using plotProjectionMovie.

        Parameters
        ----------
        sinogramNo : int,
            Must be from 0 to the total number of Y pixels number of projection
            images you took.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str,
            Colormap of choice. You can choose from the ones on
            matplotlib.colors.
        cmapRange : list with 2 entries, [0,1]
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        if len(self.prj_imgs.shape) == 3:
            imgData = self.prj_imgs[:, sinogramNo, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        if cmapRange == None:
            c_min = np.min(imgData)  # the min and max range of the sliders
            c_max = np.max(imgData)
        else:
            c_min = cmapRange[0]
            c_max = cmapRange[1]
        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # define sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)

        plt.show()

    def plotProjectionMovie(
        self, startNo=0, endNo=None, skipFrames=1, saveMovie=None, figSize=(8, 3)
    ):
        """
        Exports an animation. Run the plotProjectionImage function first to
        determine colormap range.

        Parameters
        ----------
        startNo : int
            must be from 0 to the total number of Y pixels.
        endNo : int
            must be from startNo to the total number of Y pixels.
        skipFrames : int
            number of frames you would like to skip. Increase
            this value for large datasets.
        saveMovie : 'Y', optional
            Saves movie to 'movie.mp4' (do not know if functional).
        figSize : (int, int)
            choose the figure size you want to pop out.

        Returns
        -------
        ani : matplotlib animation, can be used to plot in jupyter notebook
        using HTML(ani.to_jshtml())
        """
        frames = []
        if endNo == None:
            endNo = self.num_theta
        animSliceNos = range(startNo, endNo, skipFrames)
        fig, ax = plt.subplots(figsize=figSize)
        for i in animSliceNos:
            frames.append(
                [
                    ax.imshow(
                        self.prj_imgs[i, :, :],
                        cmap="viridis",
                        vmin=self.cbarRange[0],
                        vmax=self.cbarRange[1],
                    )
                ]
            )
            ani = animation.ArtistAnimation(
                fig, frames, interval=50, blit=True, repeat_delay=100
            )
        if saveMovie == "Y":
            ani.save("movie.mp4")
        plt.close()
        return ani

    def writeTiff(self, fname):
        """
        Writes prj_imgs attribute to file.

        Parameters
        ----------
        filename : str, relative or absolute filepath.

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


################################### Recon ###################################

################################### Recon ###################################


class Recon:
    """
    Class for performing reconstructions.

    Parameters
    ----------
    tomo : TomoData object.
        Normalize the raw tomography data with the TomoData class. Then,
        initialize this class with a TomoData object.
    """

    def __init__(
        self,
        tomo,
        center=None,
        recon=None,
        numSlices=None,
        numX=None,
        numY=None,
        options=dict(),
        alignment_time=dict(),
        recon_time=dict(),
        cbarRange=[0, 1],
        sx=None,
        sy=None,
        prjRangeX=None,
        prjRangeY=None,
    ):
        self.tomo = tomo  # tomodata object
        self.recon = recon
        self.center = tomo.numX / 2  # defaults to center of X array
        self.options = options
        self.numX = numX
        self.numY = numY
        self.numSlices = numSlices
        self.recon_time = recon_time
        self.alignment_time = alignment_time
        self.sx = sx
        self.sy = sy
        self.prjRangeX = prjRangeX
        self.prjRangeY = prjRangeY
        self.cbarRange = cbarRange

    # --------------------------Astra Reconstruction----------------------#

    def reconstructAstra(
        self,
        options={
            "proj_type": "cuda",
            "method": "SIRT_CUDA",
            "num_iter": 1,
            "ncore": 1,
            "extra_options": {"MinConstraint": 0},
        },
    ):
        """
        Reconstructs projection images after creating a Recon object.
        Uses the Astra toolbox to do the reconstruction.
        Puts the reconstructed dataset into self.recon.

        Parameters
        ----------
        options : dict
            Dictionary format typically used to specify options in tomopy, see
            LINK TO DESCRIPTION OF OPTIONS.

        """
        import astra

        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        if "center" in options:
            self.center = options["center"]
        else:
            options["center"] = self.center
        # Perform the reconstruction
        """
        print(
            "Astra reconstruction beginning on projection images",
            self.prjRange[0],
            "to",
            self.prjRange[-1],
            ".",
        )
        """
        print(
            "Running",
            str(options["method"]),
            "for",
            str(options["num_iter"]),
            "iterations.",
        )
        if "extra_options" in options:
            print("Extra options: " + str(options["extra_options"]))
        tic = time.perf_counter()
        if self.prjRangeX == None and self.prjRangeY == None:
            self.recon = tomopy.recon(
                self.tomo.prj_imgs,
                self.tomo.theta,
                algorithm=tomopy.astra,
                options=options,
            )
        elif self.prjRangeX == None and self.prjRangeY is not None:
            self.recon = tomopy.recon(
                self.tomo.prj_imgs[:, self.prjRangeY[0] : self.prjRangeY[1] : 1, :],
                self.tomo.theta,
                algorithm=tomopy.astra,
                options=options,
            )
        elif self.prjRangeX is not None and self.prjRangeY is not None:
            self.recon = tomopy.recon(
                self.tomo.prj_imgs[
                    :,
                    self.prjRangeY[0] : self.prjRangeY[1] : 1,
                    self.prjRangeX[0] : self.prjRangeX[1] : 1,
                ],
                self.tomo.theta,
                algorithm=tomopy.astra,
                options=options,
            )

        toc = time.perf_counter()
        self.reconTime = {
            "seconds": tic - toc,
            "minutes": (tic - toc) / 60,
            "hours": (tic - toc) / 3600,
        }
        print(f"Finished reconstruction after {toc - tic:0.3f} seconds.")
        self.recon = tomopy.circ_mask(self.recon, 0)
        self.numSlices, self.numY, self.numX = self.recon.shape
        self.options = options

    # ----------------------------Astra with Cupy-----------------------------#

    # --------------------------Tomopy Reconstruction----------------------#

    def reconstructTomopy(
        self,
        options={
            "algorithm": "gridrec",
            "filter_name": "butterworth",
            "filter_par": [0.25, 2],
        },
    ):
        """
        Reconstructs projection images after creating a Recon object.
        Uses tomopy toolbox to do the reconstruction.

        Parameters
        ----------
        options : dict,
            Dictionary format typically used to specify options in tomopy, see
            LINK TO DESCRIPTION OF OPTIONS.

        """
        import os

        if "center" in options:
            self.center = options["center"]
        else:
            options["center"] = self.center
        # Beginning reconstruction on range specified.
        print(
            "Tomopy reconstruction beginning on projection images",
            self.prjRange[0],
            "to",
            self.prjRange[-1],
            ".",
        )
        print("Running", str(options["algorithm"]))
        print("Options:", str(options))
        os.environ["TOMOPY_PYTHON_THREADS"] = "40"
        tic = time.perf_counter()
        self.recon = tomopy.recon(
            self.tomo.prj_imgs[:, self.prjRange[0] : self.prjRange[-1] : 1, :],
            self.tomo.theta,
            ncore=40,
            **options,
        )
        toc = time.perf_counter()
        self.reconTime = {
            "seconds": tic - toc,
            "minutes": (tic - toc) / 60,
            "hours": (tic - toc) / 3600,
        }
        print(f"Finished reconstruction after {toc - tic:0.3f} seconds.")
        self.recon = tomopy.circ_mask(self.recon, 0)
        self.numSlices, self.numY, self.numX = self.recon.shape
        self.options = options

    # --------------------------Plotting Section----------------------#

    def plotReconSlice(self, sliceNo=0, figSize=(8, 4), cmap="viridis"):
        """
        Plot a slice of the reconstructed data. TODO: take arguments to slice
        through X, Y, or Z.

        This has controls so that you can plot the image and set the correct
        color map range. The colormap range set here can be used to plot a
        movie using plotProjectionMovie.

        Sliders idea taken from: https://stackoverflow.com/questions/65040676/
        matplotlib-sliders-rescale-colorbar-when-used-to-change-clim

        Parameters
        ----------
        sliceNo : int
            Must be from 0 to Z dimension of reconstructed object.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str
            Colormap of choice. You can choose from the ones on
            matplotlib.colors
        cmapRange : list with 2 entries, [0,1]
            TODO: FEATURE NOT ON HERE.
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        imgData = self.recon[sliceNo, :, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        c_min = np.min(imgData)  # the min and max range of the sliders
        c_max = np.max(imgData)
        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # defines sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)
        plt.show()

    def plotReconMovie(
        self, startNo=0, endNo=None, skipFrames=1, saveMovie=None, figSize=(4, 4)
    ):
        """
        Exports an animation. Run the plotReconSlice function first to
        determine colormap range.

        Parameters
        ----------
        startNo : int
            must be from 0 to the total number of Y pixels.
        endNo : int
            must be from startNo to the total number of Y pixels.
        skipFrames : int, optional
            number of frames you would like to skip. Increase
            this value for large datasets.
        saveMovie : 'Y', optional
            Saves movie to 'movie.mp4' TODO: make naming file possible.
        figSize : (int, int)
            Choose the figure size you want to pop out.

        Returns
        -------
        ani : matplotlib animation,
            can be used to plot in jupyter notebook, using
            HTML(ani.to_jshtml())
            TODO: MAKE THIS ONE TASK SO THAT JUPYTER NOTEBOOK IS CLEANER
        """
        frames = []
        if endNo == None:
            endNo = self.numSlices
        animSliceNos = range(startNo, endNo, skipFrames)
        fig, ax = plt.subplots(figsize=figSize)
        for i in animSliceNos:
            frames.append(
                [
                    ax.imshow(
                        self.recon[i, :, :],
                        cmap="viridis",
                        vmin=self.cbarRange[0],
                        vmax=self.cbarRange[1],
                    )
                ]
            )
            ani = animation.ArtistAnimation(
                fig, frames, interval=50, blit=True, repeat_delay=100
            )
        if saveMovie == "Y":
            ani.save("movie.mp4")
        plt.close()
        return ani


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
