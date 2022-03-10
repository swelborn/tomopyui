import numpy as np
from ipywidgets import *
from tomopyui._sharedvars import *
import copy
from abc import ABC, abstractmethod
from tomopyui.widgets.view import (
    BqImViewer_Import_Analysis,
    BqImViewer_Altered_Analysis,
    BqImViewer_DataExplorer,
)
from tomopyui.backend.align import TomoAlign
from tomopyui.backend.recon import TomoRecon
from tomopyui.backend.io import (
    save_metadata,
    load_metadata,
    Projections_Prenormalized_General,
)
from tomopyui.widgets import helpers
from tomopyui.widgets.imports import ShiftsUploader
from tomopyui.widgets.view import BqImViewer_Prep, BqImViewer_Altered_Prep
from tomopyui.backend.util.padding import *
from functools import partial

if os.environ["cuda_enabled"] == "True":
    from ..tomocupy.prep.alignment import shift_prj_cp

import tomopy.misc.corr as tomocorr


class Prep(ABC):
    def __init__(self, Import):
        self.init_attributes(Import)
        self.init_widgets()
        self.set_observes()
        self.make_tab()

    def init_attributes(self, Import):

        self.Import = Import
        self.Import.Prep = self
        self.imported_projections = Import.projections
        self.altered_projections = Projections_Prenormalized_General()
        self.prep_list = []
        self.metadata = {}
        self.accordions_open = False
        self.preview_only = False
        self.tomocorr_median_filter_size = 3
        self.tomocorr_gaussian_filter_order = 0
        self.tomocorr_gaussian_filter_sigma = 3

    def init_widgets(self):
        """
        Initializes widgets in the Prep tab.
        """
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
        }
        self.button_font = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")

        # -- Viewers -------------------------------------------------------------------
        self.imported_plotter = BqImViewer_Prep(self)
        self.imported_plotter.create_app()
        self.altered_plotter = BqImViewer_Altered_Prep(self.imported_plotter, self)
        self.altered_plotter.create_app()

        # -- Button to turn on tab ---------------------------------------------
        self.open_accordions_button = Button(
            icon="lock-open",
            layout=self.button_layout,
            style=self.button_font,
        )

        # -- Headers for plotting -------------------------------------
        self.import_plot_header = "Imported Projections"
        self.import_plot_header = Label(
            self.import_plot_header, style=self.header_font_style
        )
        self.altered_plot_header = "Altered Projections"
        self.altered_plot_header = Label(
            self.altered_plot_header, style=self.header_font_style
        )

        # -- Header for methods -------------------------------------
        self.prep_list_header = "Methods"
        self.prep_list_header = Label(
            self.prep_list_header, style=self.header_font_style
        )

        # -- Prep List -------------------------------------
        self.prep_list_select = Select(
            options=["Method 1", "Method 2", "Method 3", "Method 4", "Method 5"],
            rows=5,
            disabled=True,
        )
        # -- Buttons for methods list -------------------------------------
        self.up_button = Button(
            icon="arrow-up",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.down_button = Button(
            icon="arrow-down",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.remove_method_button = Button(
            icon="fa-minus-square",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.start_button = Button(
            disabled=True,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip=(
                "Run the list above. "
                + "This will save a subdirectory with your processed images."
            ),
            icon="fa-running",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.methods_button_box = HBox(
            [
                self.up_button,
                self.down_button,
                self.remove_method_button,
                self.start_button,
            ]
        )

        # -- Plotting -------------------------------------------------------------

        self.plotter_hbox = HBox(
            [
                VBox(
                    [
                        self.import_plot_header,
                        self.imported_plotter.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
                VBox(
                    [
                        self.prep_list_header,
                        self.prep_list_select,
                        self.methods_button_box,
                    ],
                    layout=Layout(align_items="center"),
                ),
                VBox(
                    [
                        self.altered_plot_header,
                        self.altered_plotter.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
            ],
            layout=Layout(justify_content="center"),
        )

        # -- Shifts uploader --------------------------------------------------------
        self.shifts_uploader = ShiftsUploader(self)
        self.shift_x_header = "Shift in X"
        self.shift_x_header = Label(self.shift_x_header, style=self.header_font_style)
        self.shifts_sx_select = Select(
            options=[],
            rows=10,
            disabled=True,
        )
        self.shift_y_header = "Shift in Y"
        self.shift_y_header = Label(self.shift_y_header, style=self.header_font_style)
        self.shifts_sy_select = Select(
            options=[],
            rows=10,
            disabled=True,
        )
        self.shifts_filechooser_label = "Filechooser"
        self.shifts_filechooser_label = Label(
            self.shifts_filechooser_label, style=self.header_font_style
        )

        # -- List manipulation ---------------------------------------------------------
        self.buttons_to_disable = [
            self.start_button,
            self.up_button,
            self.down_button,
            self.remove_method_button,
            self.start_button,
        ]

        # -- Add preprocessing steps widgets -------------------------------------------

        # tomopy.misc.corr Median Filter
        self.tomocorr_median_filter_button = Button(
            description="Median Filter",
            button_style="",
            tooltip="Add a median filter to your data.",
            icon="fa-filter",
            layout=Layout(width="auto"),
        )
        # tomopy.misc.corr Median Filter options
        self.tomocorr_median_filter_size_dd = Dropdown(
            description="Size",
            options=list((str(i), i) for i in range(1, 25, 2)),
            value=3,
        )

        self.tomocorr_median_filter_box = HBox(
            [
                self.tomocorr_median_filter_button,
                self.tomocorr_median_filter_size_dd,
            ]
        )
        # tomopy.misc.corr Gaussian Filter
        self.tomocorr_gaussian_filter_button = Button(
            description="Gaussian Filter",
            button_style="",
            tooltip="Add a gaussian filter to your data.",
            icon="fa-filter",
            layout=Layout(width="auto"),
        )
        # tomopy.misc.corr Gaussian Filter options
        self.tomocorr_gaussian_filter_sigma_tb = BoundedFloatText(
            description="σ (stdv)",
            value=3,
            min=0,
            max=25,
        )
        self.tomocorr_gaussian_filter_order_dd = Dropdown(
            description="Order",
            options=[("Zeroth", 0), ("First", 1), ("Second", 2), ("Third", 3)],
            value=0,
            min=0,
            max=25,
        )

        self.tomocorr_gaussian_filter_box = HBox(
            [
                self.tomocorr_gaussian_filter_button,
                self.tomocorr_gaussian_filter_sigma_tb,
                self.tomocorr_gaussian_filter_order_dd,
            ],
        )

        # tomopy.misc.corr Gaussian Filter options
        self.renormalize_by_roi_button = Button(
            description="Click to normalize by ROI.",
            button_style="info",
            layout=Layout(width="auto", height="auto", align_items="stretch"),
            disabled=True,
        )
        # self.renormalize_by_roi_button.on_click(self.renormalize_by_roi)

        self.prep_buttons = [
            self.tomocorr_median_filter_box,
            self.tomocorr_gaussian_filter_box,
        ]

    # -- Functions to add to list ----------------------------------------
    def add_shift(self, *args):
        method = PrepMethod(
            self,
            "Shift",
            shift_projections,
            [
                self.shifts_uploader.sx,
                self.shifts_uploader.sy,
            ],
        )
        self.prep_list.append(method.method_tuple)
        self.update_prep_list()

    def add_tomocorr_median_filter(self, *args):
        self.tomocorr_median_filter_size = self.tomocorr_median_filter_size_dd.value
        method = PrepMethod(
            self,
            "Median Filter",
            tomocorr.median_filter,
            [
                self.tomocorr_median_filter_size,
            ],
        )
        self.prep_list.append(method.method_tuple)
        self.update_prep_list()

    def add_tomocorr_gaussian_filter(self, *args):
        self.tomocorr_gaussian_filter_sigma = (
            self.tomocorr_gaussian_filter_sigma_tb.value
        )
        self.tomocorr_gaussian_filter_order = (
            self.tomocorr_gaussian_filter_order_dd.value
        )
        method = PrepMethod(
            self,
            "Gaussian Filter",
            tomocorr.gaussian_filter,
            [
                self.tomocorr_gaussian_filter_sigma,
                self.tomocorr_gaussian_filter_order,
            ],
        )
        self.prep_list.append(method.method_tuple)
        self.update_prep_list()

    def add_ROI_background(self, *args):
        method = PrepMethod(
            self,
            "ROI Normalization",
            renormalize_by_roi,
            [
                self.imported_plotter.pixel_range_x,
                self.imported_plotter.pixel_range_y,
            ],
        )
        self.prep_list.append(method.method_tuple)
        self.update_prep_list()

    def update_prep_list(self):
        if self.prep_list == []:
            self.prep_list_select.options = [
                "Method 1",
                "Method 2",
                "Method 3",
                "Method 4",
                "Method 5",
            ]
            for x in self.buttons_to_disable:
                x.disabled = True
        else:
            self.prep_list_select.options = [x[0] for x in self.prep_list]
            for x in self.buttons_to_disable:
                x.disabled = False

    def refresh_plots(self):
        self.imported_plotter.plot()

    def set_metadata(self):
        pass

    def set_observes(self):

        # Start button
        self.start_button.on_click(self.run_prep_list)

        # Shifts
        self.shifts_uploader.import_button.on_click(self.add_shift)

        # tomopy.misc.corr Median Filter
        self.tomocorr_median_filter_button.on_click(self.add_tomocorr_median_filter)

        # tomopy.misc.corr Gaussian Filter
        self.tomocorr_gaussian_filter_button.on_click(self.add_tomocorr_gaussian_filter)

    def update_shifts_list(self):
        pass

    # -- Radio to turn on tab ---------------------------------------------
    def activate_tab(self, *args):
        if self.accordions_open is False:
            self.open_accordions_button.icon = "fa-lock"
            self.open_accordions_button.button_style = "success"
            self.projections = self.Import.projections
            self.set_metadata()
            self.shifts_accordion.selected_index = 0
            self.plotter_accordion.selected_index = 0
            self.accordions_open = True
        else:
            self.open_accordions_button.icon = "fa-lock-open"
            self.open_accordions_button.button_style = "info"
            self.accordions_open = False
            self.load_metadata_button.disabled = True
            self.shifts_accordion.selected_index = None
            self.plotter_accordion.selected_index = None

    # -- Load metadata button ---------------------------------------------
    def _load_metadata_all_on_click(self, change):
        self.load_metadata_button.button_style = "info"
        self.load_metadata_button.icon = "fas fa-cog fa-spin fa-lg"
        self.load_metadata_button.description = "Importing metadata."
        self.load_metadata_button.button_style = "success"
        self.load_metadata_button.icon = "fa-check-square"
        self.load_metadata_button.description = "Finished importing metadata."

    # -- Button to start alignment ----------------------------------------
    def run_prep_list(self, change):
        change.button_style = "info"
        change.icon = "fas fa-cog fa-spin fa-lg"
        self.run()
        change.button_style = "success"
        change.icon = "fa-check-square"

    def run(self):
        self.first_method = True
        if not self.preview_only:
            self.prepped_data = copy.deepcopy(self.imported_plotter.original_imagestack)
        for prep_method_tuple in self.prep_list:
            prep_method_tuple[1].update_method_and_run()
            self.first_method = False
        if not self.preview_only:
            self.altered_plotter.plot()

    def make_tab(self):

        # -- Box organization -------------------------------------------------

        self.top_of_box_hb = HBox(
            [self.open_accordions_button, self.Import.switch_data_buttons],
            layout=Layout(
                width="auto",
                justify_content="flex-start",
            ),
        )
        self.plotter_accordion = Accordion(
            children=[self.plotter_hbox],
            selected_index=0,
            titles=("Plot Projection Images",),
        )
        self.prep_buttons_hbox = VBox(
            self.prep_buttons,
            layout=Layout(justify_content="center"),
        )
        self.prep_buttons_accordion = Accordion(
            children=[self.prep_buttons_hbox],
            selected_index=0,
            titles=("Add Preprocessing Methods",),
        )

        self.shifts_box = HBox(
            [
                VBox(
                    [
                        self.shifts_uploader.quick_path_label,
                        self.shifts_uploader.quick_path_search,
                    ],
                ),
                VBox(
                    [
                        self.shifts_filechooser_label,
                        self.shifts_uploader.filechooser,
                    ],
                ),
                VBox(
                    [
                        self.shift_x_header,
                        self.shifts_sx_select,
                    ],
                ),
                VBox(
                    [
                        self.shift_y_header,
                        self.shifts_sy_select,
                    ],
                ),
                self.shifts_uploader.import_button,
            ],
            layout=Layout(justify_content="center"),
        )

        self.shifts_accordion = Accordion(
            children=[self.shifts_box],
            selected_index=None,
            titles=("Upload shifts from prior alignments",),
        )

        # progress_hbox = HBox(
        #     [
        #         self.progress_total,
        #         self.progress_reprj,
        #         self.progress_phase_cross_corr,
        #         self.progress_shifting,
        #     ],
        #     layout=Layout(justify_content="center"),
        # )

        self.tab = VBox(
            children=[
                self.top_of_box_hb,
                self.plotter_accordion,
                self.prep_buttons_accordion,
                self.shifts_accordion,
            ]
        )


class PrepMethod:
    def __init__(self, Prep, method_name: str, func, opts: list):
        self.Prep = Prep
        self.method_name = method_name
        self.func = func
        self.opts = opts
        self.method_tuple = (self.method_name, self)

    def update_method_and_run(self):
        if self.Prep.preview_only:
            if self.Prep.first_method:
                image_index = self.Prep.imported_plotter.image_index_slider.value
                data = self.Prep.imported_projections[image_index]
                data = data[np.newaxis, ...]
            else:
                data = self.Prep.preview_image
            self.partial_func = partial(self.func, data, *self.opts)
            self.Prep.preview_image = self.partial_func()
            self.Prep.altered_plotter.plotted_image.image = self.Prep.preview_image[0]
        else:
            data = self.Prep.prepped_data
            self.partial_func = partial(self.func, data, *self.opts)
            self.Prep.prepped_data = self.partial_func()


def shift_projections(projections, sx, sy):
    new_prj_imgs = copy.deepcopy(projections)
    pad_x = np.max(np.abs(sx))
    pad_y = np.max(np.abs(sy))
    pad = (pad_x, pad_y)
    new_prj_imgs, pad = pad_projections(new_prj_imgs, pad)
    new_prj_imgs = shift_prj_cp(
        new_prj_imgs,
        sx,
        sy,
        5,
        pad,
        use_corr_prj_gpu=False,
    )
    return new_prj_imgs


def shift_projections(projections, sx, sy):
    new_prj_imgs = copy.deepcopy(projections)
    pad_x = np.max(np.abs(sx))
    pad_y = np.max(np.abs(sy))
    pad = (pad_x, pad_y)
    new_prj_imgs, pad = pad_projections(new_prj_imgs, pad)
    new_prj_imgs = shift_prj_cp(
        new_prj_imgs,
        sx,
        sy,
        5,
        pad,
        use_corr_prj_gpu=False,
    )
    return new_prj_imgs


def renormalize_by_roi(projections, px_range_x, px_range_y):
    exp_full = np.exp(-projections)
    averages = [
        np.mean(
            exp_full[i, px_range_y[0] : px_range_y[1], px_range_x[0] : px_range_x[1]]
        )
        for i in range(len(exp_full))
    ]
    prj = [exp_full[i] / averages[i] for i in range(len(exp_full))]
    prj = -np.log(prj)
    return prj
