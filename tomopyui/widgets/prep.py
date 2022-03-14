import numpy as np
import copy
import datetime
import pathlib
from ipywidgets import *
from abc import ABC, abstractmethod
from functools import partial
from tomopyui._sharedvars import *
from tomopyui.backend.align import TomoAlign
from tomopyui.backend.recon import TomoRecon
from tomopyui.backend.io import Projections_Prenormalized
from tomopyui.widgets.imports import ShiftsUploader, TwoEnergyUploader
from tomopyui.widgets.view import (
    BqImViewer_Prep,
    BqImViewer_Altered_Prep,
    BqImViewer_TwoEnergy_High,
    BqImViewer_TwoEnergy_Low,
)
from tomopyui.backend.util.padding import *
from tomopyui.backend.io import Metadata_Prep


if os.environ["cuda_enabled"] == "True":
    from ..tomocupy.prep.alignment import shift_prj_cp, batch_cross_correlation
    from ..tomocupy.prep.sampling import shrink_projections

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
        self.altered_projections = Projections_Prenormalized()
        self.prep_list = []
        self.metadata = {}
        self.accordions_open = False
        self.preview_only = False
        self.tomocorr_median_filter_size = 3
        self.tomocorr_gaussian_filter_order = 0
        self.tomocorr_gaussian_filter_sigma = 3
        self.save_on = False
        self.metadata = Metadata_Prep()

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

        # -- Main viewers --------------------------------------------------------------
        self.imported_viewer = BqImViewer_Prep(self)
        self.imported_viewer.create_app()
        self.altered_viewer = BqImViewer_Altered_Prep(self.imported_viewer, self)
        self.altered_viewer.create_app()

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
            rows=10,
            disabled=True,
        )
        # -- Buttons for methods list -------------------------------------
        self.up_button = Button(
            disabled=True,
            icon="arrow-up",
            tooltip="Move method up.",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.down_button = Button(
            disabled=True,
            icon="arrow-down",
            tooltip="Move method down.",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.remove_method_button = Button(
            disabled=True,
            icon="fa-minus-square",
            tooltip="Remove selected method.",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.start_button = Button(
            disabled=True,
            button_style="info",
            tooltip=(
                "Run the list above. "
                + "This will save a subdirectory with your processed images."
            ),
            icon="fa-running",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.preview_only_button = Button(
            disabled=True,
            button_style="",
            tooltip=(
                "Run the currently selected image through your list of methods. "
                + "This will not run the stack or save data."
            ),
            icon="glasses",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.save_on_button = Button(
            disabled=True,
            button_style="",
            tooltip=("Turn this on to save the data when you click the run button."),
            icon="fa-file-export",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.methods_button_box = VBox(
            [
                HBox(
                    [
                        self.up_button,
                        self.down_button,
                        self.remove_method_button,
                    ]
                ),
                HBox(
                    [
                        self.preview_only_button,
                        self.start_button,
                        self.save_on_button,
                    ]
                ),
            ]
        )

        # -- Main Viewers -------------------------------------------------------------

        self.viewer_hbox = HBox(
            [
                VBox(
                    [
                        self.import_plot_header,
                        self.imported_viewer.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
                VBox(
                    [
                        self.prep_list_header,
                        self.prep_list_select,
                        self.methods_button_box,
                    ],
                    layout=Layout(align_items="center", align_content="center"),
                ),
                VBox(
                    [
                        self.altered_plot_header,
                        self.altered_viewer.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
            ],
            layout=Layout(justify_content="center", align_items="center"),
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
            self.prep_list_select,
            self.start_button,
            self.up_button,
            self.down_button,
            self.remove_method_button,
            self.start_button,
            self.preview_only_button,
            self.save_on_button,
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

        # -- Widgets for shifting other energies tool ----------------------------------
        self.high_e_viewer = BqImViewer_TwoEnergy_High()
        self.low_e_viewer = BqImViewer_TwoEnergy_Low(self.high_e_viewer)
        self.high_e_uploader = TwoEnergyUploader(self.high_e_viewer)
        self.low_e_uploader = TwoEnergyUploader(self.low_e_viewer)
        self.high_e_header = "Shifted High Energy Projections"
        self.high_e_header = Label(self.high_e_header, style=self.header_font_style)
        self.low_e_header = "Moving Low Energy Projections"
        self.low_e_header = Label(self.low_e_header, style=self.header_font_style)
        self.low_e_viewer.scale_button.on_click(self.scale_low_e)
        self.num_batches_textbox = IntText(description="Number of batches: ", value=5)
        self.two_e_shift_uploaders_hbox = HBox(
            [
                VBox(
                    [
                        self.high_e_uploader.quick_path_label,
                        HBox(
                            [
                                self.high_e_uploader.quick_path_search,
                                self.high_e_uploader.import_button,
                            ]
                        ),
                        self.high_e_uploader.filechooser,
                    ],
                ),
                VBox(
                    [
                        self.low_e_uploader.quick_path_label,
                        HBox(
                            [
                                self.low_e_uploader.quick_path_search,
                                self.low_e_uploader.import_button,
                            ]
                        ),
                        self.low_e_uploader.filechooser,
                    ],
                ),
            ],
            layout=Layout(justify_content="center"),
        )

        self.two_e_shift_viewer_hbox = HBox(
            [
                VBox(
                    [
                        self.high_e_header,
                        self.high_e_viewer.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
                VBox(
                    [
                        self.low_e_header,
                        self.low_e_viewer.app,
                        self.num_batches_textbox,
                    ],
                    layout=Layout(align_items="center"),
                ),
            ],
            layout=Layout(justify_content="center", align_items="center"),
        )

        self.two_e_shift_box = VBox(
            [self.two_e_shift_uploaders_hbox, self.two_e_shift_viewer_hbox]
        )

    # -- Functions for Energy Scaling/Shifting ----------------------------
    def scale_low_e(self, *args):
        low_e = self.low_e_viewer.projections.energy
        high_e = self.high_e_viewer.projections.energy
        num_batches = self.num_batches_textbox.value
        high_e_prj = self.high_e_viewer.projections.data
        self.low_e_viewer.scale_button.button_style = "info"
        self.low_e_viewer.scale_button.icon = "fas fa-cog fa-spin fa-lg"
        self.low_e_viewer.projections.data = shrink_and_pad_projections(
            self.low_e_viewer.projections.data, high_e_prj, high_e, low_e, num_batches
        )
        self.low_e_viewer.plot(self.low_e_viewer.projections)
        self.low_e_viewer.start_button.disabled = False
        self.low_e_viewer.scale_button.button_style = "success"
        self.low_e_viewer.scale_button.icon = "fa-check-square"

    def register_low_e(self, *args):
        low_range_x = self.low_e_viewer.pixel_range_x
        low_range_y = self.low_e_viewer.pixel_range_y
        high_range_x = self.high_e_viewer.pixel_range_x
        high_range_x = self.high_e_viewer.pixel_range_y
        self.low_e_viewer.start_button.button_style = "info"
        self.low_e_viewer.start_button.icon = "fas fa-cog fa-spin fa-lg"
        num_batches = self.num_batches_textbox.value
        upsample_factor = 50
        shift_cpu = []
        low_e_data = self.low_e_viewer.projections.data[
            :, low_range_y[0] : low_range_y[1], low_range_x[0] : low_range_x[1]
        ]
        high_e_data = self.high_e_viewer.projections.data[
            :, high_range_y[0] : high_range_y[1], high_range_x[0] : high_range_x[1]
        ]

        batch_cross_correlation(
            low_e_data,
            high_e_data,
            shift_cpu,
            num_batches,
            upsample_factor,
            blur=False,
            subset_correlation=False,
            subset_x=None,
            subset_y=None,
            mask_sim=False,
            pad=(0, 0),
            progress=None,
        )
        shift_cpu = np.concatenate(shift_cpu, axis=1)
        sx = shift_cpu[1]
        sy = shift_cpu[0]
        # TODO: send to GPU and do both calcs there.
        self.low_e_viewer.projections.data = shift_prj_cp(
            prj, sx, sy, num_batches, (0, 0), use_pad_cond=False, use_corr_prj_gpu=False
        )
        self.low_e_viewer.plot(self.low_e_viewer.projections)
        self.low_e_viewer.start_button.button_style = "success"
        self.low_e_viewer.start_button.icon = "fa-check-square"

    # -- Functions to add to list ----------------------------------------
    def add_shift(self, *args):
        method = PrepMethod(
            self,
            "Shift",
            shift_projections,
            [
                list(self.shifts_uploader.sx),
                list(self.shifts_uploader.sy),
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
                self.imported_viewer.pixel_range_x,
                self.imported_viewer.pixel_range_y,
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
        self.imported_projections = self.Import.projections
        self.imported_viewer.plot()

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

        # Remove method
        self.remove_method_button.on_click(self.remove_method)

        # Move method up
        self.up_button.on_click(self.move_method_up)

        # Move method down
        self.down_button.on_click(self.move_method_down)

        # Preview
        self.preview_only_button.on_click(self.preview_only_on_off)

        # Save
        self.save_on_button.on_click(self.save_on_off)

    def update_shifts_list(self):
        pass

    def save_on_off(self, *args):
        if self.save_on:
            self.save_on_button.button_style = ""
            self.save_on = False
        else:
            self.save_on_button.button_style = "success"
            self.save_on = True

    def preview_only_on_off(self, *args):
        if self.preview_only:
            self.preview_only_button.button_style = ""
            self.preview_only = False
        else:
            self.preview_only_button.button_style = "success"
            self.preview_only = True

    def remove_method(self, *args):
        ind = self.prep_list_select.index
        self.prep_list.pop(ind)
        self.update_prep_list()

    def move_method_up(self, *args):
        ind = self.prep_list_select.index
        if ind != 0:
            self.prep_list[ind], self.prep_list[ind - 1] = (
                self.prep_list[ind - 1],
                self.prep_list[ind],
            )
            self.update_prep_list()
            self.prep_list_select.index = ind - 1

    def move_method_down(self, *args):
        ind = self.prep_list_select.index
        if ind != len(self.prep_list) - 1:
            self.prep_list[ind], self.prep_list[ind + 1] = (
                self.prep_list[ind + 1],
                self.prep_list[ind],
            )
            self.update_prep_list()
            self.prep_list_select.index = ind + 1

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
        if self.preview_only:
            image_index = self.imported_viewer.image_index_slider.value
            self.altered_viewer.image_index_slider.value = image_index
            self.prepped_data = copy.deepcopy(
                self.imported_viewer.original_imagestack[image_index]
            )
            self.prepped_data = self.prepped_data[np.newaxis, ...]
            for prep_method_tuple in self.prep_list:
                prep_method_tuple[1].update_method_and_run()
            self.altered_viewer.plotted_image.image = self.prepped_data[0]
        else:
            self.prepped_data = copy.deepcopy(self.imported_viewer.original_imagestack)
            for num, prep_method_tuple in enumerate(self.prep_list):
                prep_method_tuple[1].update_method_and_run()
                self.prep_list_select.index = num
            self.altered_viewer.original_imagestack = self.prepped_data
            self.altered_viewer.plot()
            if self.save_on:
                self.make_prep_dir()
                self.metadata.set_metadata(self)
                self.metadata.filedir = self.filedir
                self.metadata.save_metadata()
                self.metadata.parent_metadata = self.filedir
                self.metadata.parent_metadata.save_metadata()
                self.altered_projections.data = self.prepped_data
                np.save(self.filedir / "prepped_projections.npy", self.prepped_data)

    def make_prep_dir(self):
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S-prep")
        self.filedir = pathlib.Path(self.Import.projections.filedir) / dt_string
        os.mkdir(self.filedir)

    def make_tab(self):

        # -- Box organization -------------------------------------------------

        self.top_of_box_hb = HBox(
            [self.Import.switch_data_buttons],
            layout=Layout(
                width="auto",
                justify_content="flex-start",
            ),
        )
        self.viewer_accordion = Accordion(
            children=[self.viewer_hbox],
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
        self.two_e_shift_accordion = Accordion(
            children=[self.two_e_shift_box],
            selected_index=0,
            titles=("Tool: shift projections.",),
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
                self.viewer_accordion,
                self.prep_buttons_accordion,
                self.shifts_accordion,
                self.two_e_shift_accordion,
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
        self.partial_func = partial(self.func, self.Prep.prepped_data, *self.opts)
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
    pad_x = int(np.ceil(np.max(np.abs(sx))))
    pad_y = int(np.ceil(np.max(np.abs(sy))))
    pad = (pad_x, pad_y)
    new_prj_imgs, pad = pad_projections(new_prj_imgs, pad)
    new_prj_imgs = shift_prj_cp(
        new_prj_imgs,
        sx,
        sy,
        20,
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


### May use?
# def rectangle_selector_on(self, change):
#     time.sleep(0.1)
#     if self.viewer.rectangle_selector_on:
#         self.renormalize_by_roi_button.disabled = False
#     else:
#         self.renormalize_by_roi_button.disabled = True
