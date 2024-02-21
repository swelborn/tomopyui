import logging
from abc import ABC, abstractmethod

from ipywidgets import *

from tomopyui.widgets import helpers
from tomopyui.widgets.helpers import ReactiveTextButton


class ImportBase(ABC):
    """
    An overarching class that controls the rest of the processing pipeline.
    Holds `Uploader` instances, which can be used for uploading data in the form of
    `Projections` instances. The prenorm_uploader is general, in that it can be used for
    any type of data. This is why it is in the base class. The subclasses of ImportBase
    are for creating and holding raw `Uploader` instances.
    """

    def __init__(self):

        # Init raw/prenorm button switches. These are at the top of each of the Prep,
        # Alignment, and Recon tabs to switch between using raw/uploaded data or
        # prenormalized data. See the `ReactiveButton` helper class.
        self.use_raw_button = ReactiveTextButton(
            self.enable_raw,
            "Click to use raw/normalized data from the Import tab.",
            "Updating plots.",
            "Raw/normalized data from Import tab in use for alignment/reconstruction.",
        )
        self.use_prenorm_button = ReactiveTextButton(
            self.enable_prenorm,
            "Click to use prenormalized data from the Import tab.",
            "Updating plots.",
            "Prenormalized data from Import tab in use for alignment/reconstruction.",
        )

        # Creates the prenormalized uploader (general)
        # raw_uploader is created in the beamline-specific subclasses.
        # Initializes to setting the `Import` instance's projections to be the prenorm
        # projections, but this is switched with the "enable_raw" or "disable_raw"
        # functions and buttons, defined above. Maybe I am getting terminology incorrect
        # but this is kind of like a switchable singleton.
        self.prenorm_uploader = PrenormUploader(self)
        self.projections = self.prenorm_uploader.projections
        self.uploader = self.prenorm_uploader

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

    def enable_prenorm(self, *args):
        """
        Makes the prenorm_uploader projections the projections used throughout the app.
        Refreshes plots in the other tabs to match these.
        """
        self.use_raw_button.reset_state()
        self.use_raw = False
        self.use_prenorm = True
        if self.raw_uploader.projections.hdf_file is not None:
            self.raw_uploader.projections._close_hdf_file()
        self.projections = self.prenorm_uploader.projections
        if self.projections.hdf_file is not None:
            self.projections._open_hdf_file_read_only()
            self.projections._load_hdf_ds_data_into_memory()
        # self.projections._check_downsampled_data()
        self.uploader = self.prenorm_uploader
        self.Prep.projections = self.projections
        self.Center.projections = self.projections
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.Center.refresh_plots()
        self.Prep.refresh_plots()
        self.projections._close_hdf_file()

    def enable_raw(self, *args):
        """
        Makes the raw_uploader projections the projections used throughout the app.
        Refreshes plots in the other tabs to match these.
        """
        self.use_prenorm_button.reset_state()
        self.use_raw = True
        self.use_prenorm = False
        if self.prenorm_uploader.projections.hdf_file is not None:
            self.prenorm_uploader.projections._close_hdf_file()
        self.projections = self.raw_uploader.projections
        if self.projections.hdf_file is not None:
            self.projections._open_hdf_file_read_only()
            self.projections._load_hdf_ds_data_into_memory()
        self.projections._check_downsampled_data()
        self.uploader = self.raw_uploader
        self.Prep.projections = self.projections
        self.Center.projections = self.projections
        self.Recon.projections = self.projections
        self.Align.projections = self.projections
        self.Recon.refresh_plots()
        self.Align.refresh_plots()
        self.Center.refresh_plots()
        self.Prep.refresh_plots()
        self.projections._close_hdf_file()

    @abstractmethod
    def make_tab(self): ...
