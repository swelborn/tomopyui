from tomopyui.widgets._import import import_helpers
from tomopyui.widgets._shared import helpers
from ipyfilechooser import FileChooser
import logging
from ipywidgets import *
from abc import ABC, abstractmethod
from tomopyui._sharedvars import *
import time
from tomopyui.widgets.meta import Plotter, DataExplorer
from tomopyui.widgets.plot import BqImPlotter_ImportRaw
from tomopyui.backend.io import ProjectionsXRM_SSRL62


class ImportBase(ABC):
    def __init__(self):

        # Init textboxes
        self.angle_start = -90.0
        self.angle_end = 90.0
        self.num_theta = 360
        self.prj_shape = None
        self.prj_range_x = (0, 100)
        self.prj_range_y = (0, 100)
        self.prj_range_z = (0, 100)
        self.tomo = None
        self.angles_textboxes = import_helpers.create_angles_textboxes(self)

        # Init checkboxes
        self.import_opts_list = ["rotate"]
        self.import_opts = {key: False for key in self.import_opts_list}
        self.opts_checkboxes = helpers.create_checkboxes_from_opt_list(
            self.import_opts_list, self.import_opts, self
        )

        # Init filechooser
        self.fpath = None
        self.fname = None
        self.ftype = None
        self.filechooser = FileChooser()
        self.filechooser.register_callback(self.update_file_information)
        self.filechooser.title = "Import Normalized Tomogram:"
        self.wd = None

        # Init filechooser for align metadata
        self.fpath_align = None
        self.fname_align = None
        self.filechooser_align = FileChooser()
        self.filechooser_align.register_callback(self.update_file_information_align)
        self.filechooser_align.title = "Import Alignment Metadata:"

        # Init filechooser for recon metadata
        self.fpath_recon = None
        self.fname_recon = None
        self.filechooser_recon = FileChooser()
        self.filechooser_recon.register_callback(self.update_file_information_recon)
        self.filechooser_recon.title = "Import Reconstruction Metadata:"

        # Init logger to be used throughout the app.
        # TODO: This does not need to be under Import.
        self.log = logging.getLogger(__name__)
        self.log_handler, self.log = helpers.return_handler(self.log, logging_level=20)

        # Init metadata
        self.metadata = {}
        self.set_metadata()

    def set_wd(self, wd):
        """
        Sets the current working directory of `Import` class and changes the
        current directory to it.
        """
        self.wd = wd
        os.chdir(wd)

    def set_metadata(self):
        """
        Sets relevant metadata for `Import`
        """
        self.metadata = {
            "fpath": self.fpath,
            "fname": self.fname,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
            "num_theta": self.num_theta,
            "prj_range_x": self.prj_range_x,
            "prj_range_y": self.prj_range_y,
        } | self.import_opts

    def update_file_information(self):
        """
        Callback for `Import`.filechooser.
        """
        self.fpath = self.filechooser.selected_path
        self.fname = self.filechooser.selected_filename
        self.set_wd(self.fpath)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.set_metadata()
        self.get_prj_shape()
        self.set_prj_ranges()
        self.set_metadata()

    def update_file_information_align(self):
        """
        Callback for filechooser_align.
        """
        self.fpath_align = self.filechooser_align.selected_path
        self.fname_align = self.filechooser_align.selected_filename
        self.set_metadata()

    def update_file_information_recon(self):
        """
        Callback for filechooser_recon.
        """
        self.fpath_recon = self.filechooser_recon.selected_path
        self.fname_recon = self.filechooser_recon.selected_filename
        self.set_metadata()

    def get_prj_shape(self):
        """
        Grabs the image shape depending on the filename. Does this without
        loading the image into memory.
        """
        if self.fname.__contains__(".tif"):
            self.prj_shape = helpers.get_img_shape(
                self.fpath,
                self.fname,
                "tiff",
                self.metadata,
            )
        elif self.fname.__contains__(".npy"):
            self.prj_shape = helpers.get_img_shape(
                self.fpath,
                self.fname,
                "npy",
                self.metadata,
            )
        elif self.fname == "":
            self.prj_shape = helpers.get_img_shape(
                self.fpath, self.fname, "tiff", self.metadata, folder_import=True
            )

    def set_prj_ranges(self):
        self.prj_range_x = (0, self.prj_shape[2] - 1)
        self.prj_range_y = (0, self.prj_shape[1] - 1)
        self.prj_range_z = (0, self.prj_shape[0] - 1)

    @abstractmethod
    def make_tab(self):
        ...


#     def make_tomo(self):
#         """
#         Creates a `~tomopyui.backend.tomodata.TomoData` object and stores it in
#         `Import`.

#         .. code-block:: python

#             # In Jupyter:

#             # Cell 1:
#             from ipywidgets import *
#             from tomopyui.widgets.meta import Import
#             from
#             a = Import()
#             a.tab
#             # You should see the HBox widget, you can select your file.

#             # Cell2:
#             a.make_tomo() # creates tomo.TomoData based on inputs
#             a.tomo.prj_imgs # access the projections like so.

#         """
#         self.tomo = td.TomoData(metadata=self.metadata)


class Import_SSRL62(ImportBase):
    def __init__(self):
        super().__init__()
        # Init normalizing folder chooser
        self.fpath_raw = None
        self.fname_raw = None
        self.filechooser_raw = FileChooser()
        self.filechooser_raw.register_callback(self.update_raw_file_information)
        self.filechooser_raw.title = "Import Raw XRM Folder"
        self.wd_raw = None
        self.angles_from_filenames = True
        self._init_widgets()
        self.projections = ProjectionsXRM_SSRL62()
        self.projections.set_options_from_frontend(self)
        self.DataExplorer = DataExplorer()
        self.make_tab()

    def _init_widgets(self):
        self.metadata_table_output = Output()
        self.import_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="",
            layout=Layout(width="75px", height="86px"),
            disabled=True,
        )
        self.import_button.on_click(self.import_and_normalize_onclick)
        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )
        self.progress_output = Output()
        self.upload_progress = IntProgress(
            description="Uploading: ",
            value=0,
            min=0,
            max=100,
            layout=Layout(justify_content="center"),
        )
        self.quick_path_label = Label("Quick path search")
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.bq_plotter = BqImPlotter_ImportRaw(dimensions=("100px", "100px"))
        self.bq_plotter.create_app()

    def import_and_normalize_onclick(self, change):
        tic = time.perf_counter()
        self.import_button.button_style = "info"
        self.import_button.icon = "fas fa-cog fa-spin fa-lg"
        self.progress_output.clear_output()
        self.upload_progress.value = 0
        self.upload_progress.max = self.projections.pxZ + len(
            self.projections.flats_ind
        )
        with self.progress_output:
            display(self.upload_progress)
        self.projections.import_folder_all(self.fpath_raw)
        with self.progress_output:
            display(Label("Normalizing", layout=Layout(justify_content="center")))

        self.projections.normalize_nf()
        toc = time.perf_counter()
        self.import_button.button_style = "success"
        self.import_button.icon = "fa-check-square"

        with self.progress_output:
            display(
                Label(
                    f"Import and normalization took {toc-tic:.0f}s",
                    layout=Layout(justify_content="center"),
                )
            )
            # print(f"Import and normalization took {toc-tic:.0f}s")

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.fpath_raw = path
        self.filechooser_raw.reset(path=path)
        textfiles = self.projections._file_finder(path, [".txt"])
        if textfiles == []:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have any .txt files, please try another one."
                )
            return
        scan_info_filepath = (
            path / [file for file in textfiles if "ScanInfo" in file][0]
        )
        if scan_info_filepath != []:
            self.projections.import_folder_metadata(path)
            self.metadata_table = self.projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
        else:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have a ScanInfo file, please try another one."
                )

    def update_raw_file_information(self):

        self.fpath_raw = pathlib.Path(self.filechooser_raw.selected_path)
        self.fname_raw = self.filechooser_raw.selected_filename
        self.set_wd(self.fpath_raw)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.projections.import_folder_metadata(self.fpath_raw)
        self.metadata_table = self.projections.metadata_to_DataFrame()
        self.quick_path_search.value = str(self.fpath_raw)
        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False

        # self.set_metadata()
        # self.get_prj_shape()
        # self.set_prj_ranges()
        # self.set_metadata()

    def make_tab(self):
        """
        Creates the HBox which stores widgets.
        """
        raw_import = HBox(
            [
                VBox(
                    [
                        self.quick_path_label,
                        HBox([self.quick_path_search, self.import_button]),
                        self.filechooser_raw,
                    ],
                ),
                self.bq_plotter.fig,
            ]
        )

        # raw_import = HBox([item for sublist in raw_import for item in sublist])
        raw_import_1 = Accordion(
            children=[
                VBox(
                    [
                        HBox(
                            [self.metadata_table_output],
                            layout=Layout(justify_content="center"),
                        ),
                        HBox(
                            [self.progress_output],
                            layout=Layout(justify_content="center"),
                        ),
                        raw_import,
                    ]
                ),
            ],
            selected_index=None,
            titles=("Import and Normalize Raw Data",),
        )
        norm_import = [[self.filechooser], self.angles_textboxes]
        norm_import = HBox(
            [item for sublist in norm_import for item in sublist],
            layout=Layout(justify_content="center"),
        )
        norm_import = Accordion(
            children=[norm_import],
            selected_index=None,
            titles=("Import Pre-normalized Data",),
        )
        meta_import = Accordion(
            children=[
                HBox(
                    [self.filechooser_align, self.filechooser_recon],
                    layout=Layout(justify_content="center"),
                )
            ],
            selected_index=None,
            titles=("Import Alignment/Reconstruction Settings",),
        )
        self.tab = VBox(
            [
                raw_import_1,
                norm_import,
                meta_import,
            ]
        )
