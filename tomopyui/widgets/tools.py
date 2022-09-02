# Miscellaneous Tools

import numpy as np
import copy
import datetime
import pathlib
import dask.array as da

from ipywidgets import *
from tomopyui._sharedvars import *
from tomopyui.backend.io import Projections_Child

from tomopyui.widgets.imports.imports import TwoEnergyUploader
from tomopyui.widgets.view import (
    BqImViewer_TwoEnergy_High,
    BqImViewer_TwoEnergy_Low,
)
from tomopyui.backend.util.padding import *
from tomopyui.backend.io import Metadata_TwoE
from tomopyui.widgets.helpers import ReactiveIconButton
if os.environ["cuda_enabled"] == "True":
    from ..tomocupy.prep.alignment import shift_prj_cp, batch_cross_correlation
    from ..tomocupy.prep.sampling import shrink_and_pad_projections
    from tomopyui.widgets.prep import shift_projections
# Multiple Energy Alignment

class TwoEnergyTool:

    def __init__(self):
        self.init_attributes()
        self.init_widgets()
        self.set_observes()        
        
    def init_attributes(self):
        self.metadata = Metadata_TwoE()

    def init_widgets(self):
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
        }
        self.button_font = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")
        self.high_e_header = "Shifted High Energy Projections"
        self.high_e_header = Label(self.high_e_header, style=self.header_font_style)
        self.low_e_header = "Moving Low Energy Projections"
        self.low_e_header = Label(self.low_e_header, style=self.header_font_style)
        self.high_e_viewer = BqImViewer_TwoEnergy_High()
        self.low_e_viewer = BqImViewer_TwoEnergy_Low(self.high_e_viewer)
        self.low_e_viewer.scale_button.on_click(self.scale_low_e)
        self.high_e_uploader = TwoEnergyUploader(self.high_e_viewer)
        self.low_e_uploader = TwoEnergyUploader(self.low_e_viewer)
        self.num_batches_textbox = IntText(description="Number of batches: ", value=5, style=extend_description_style)
        self.two_e_shift_uploaders_hbox = HBox(
            [
                VBox(
                    [
                        self.high_e_uploader.quick_path_label,
                        HBox(
                            [
                                self.high_e_uploader.quick_path_search,
                                self.high_e_uploader.import_button.button,
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
                                self.low_e_uploader.import_button.button,
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


    def set_observes(self):
        # Save
        self.low_e_viewer.save_button.on_click(self.save_shifted_projections)

        # Registration
        self.low_e_viewer.start_button.on_click(self.register_low_e)

    # -- Functions for Energy Scaling/Shifting ----------------------------
    def scale_low_e(self, *args):
        self.low_e_viewer.projections.metadata.set_attributes_from_metadata(
            self.low_e_viewer.projections
        )
        self.high_e_viewer.projections.metadata.set_attributes_from_metadata(
            self.high_e_viewer.projections
        )
        low_e = self.low_e_viewer.projections.energy_float
        high_e = self.high_e_viewer.projections.energy_float
        self.num_batches = self.num_batches_textbox.value
        high_e_prj = self.high_e_viewer.projections.data
        self.low_e_viewer.scale_button.button_style = "info"
        self.low_e_viewer.scale_button.icon = "fas fa-cog fa-spin fa-lg"
        self.low_e_viewer.projections.shrunken_data = shrink_and_pad_projections(
            self.low_e_viewer.projections.data, high_e_prj, low_e, high_e, self.num_batches
        )
        print("got here")
        print(self.low_e_viewer.projections.shrunken_data.shape)
        self.low_e_viewer.plot_shrunken()
        self.low_e_viewer.start_button.disabled = False
        self.low_e_viewer.scale_button.button_style = "success"
        self.low_e_viewer.scale_button.icon = "fa-check-square"
        self.low_e_viewer.diff_images = np.array(
            [x / np.mean(x) for x in self.low_e_viewer.viewer_parent.original_images]
        ) - np.array([x / np.mean(x) for x in self.low_e_viewer.original_images])
        self.low_e_viewer.diff_on = False
        self.low_e_viewer._disable_diff_callback = True
        self.low_e_viewer.diff_button.disabled = False
        self.low_e_viewer._disable_diff_callback = False

    def register_low_e(self, *args):
        self.high_range_x = self.high_e_viewer.px_range_x
        self.high_range_y = self.high_e_viewer.px_range_y
        self.low_range_x = self.low_e_viewer.px_range_x
        self.low_range_y = self.low_e_viewer.px_range_y
        self.low_range_x[1] = int(self.low_range_x[0] + (self.high_range_x[1] - self.high_range_x[0]))
        self.low_range_y[1] = int(self.low_range_y[0] + (self.high_range_y[1] - self.high_range_y[0]))
        self.low_e_viewer.start_button.button_style = "info"
        self.low_e_viewer.start_button.icon = "fas fa-cog fa-spin fa-lg"
        self.num_batches = self.num_batches_textbox.value
        self.upsample_factor = 50
        self.shift_cpu = []
        low_e_data = self.low_e_viewer.projections.shrunken_data[
            :, self.low_range_y[0] : self.low_range_y[1], self.low_range_x[0] : self.low_range_x[1]
        ]
        high_e_data = self.high_e_viewer.projections.data[
            :, self.high_range_y[0] : self.high_range_y[1], self.high_range_x[0] : self.high_range_x[1]
        ]

        batch_cross_correlation(
            low_e_data,
            high_e_data,
            self.shift_cpu,
            self.num_batches,
            self.upsample_factor,
            blur=False,
            subset_correlation=False,
            subset_x=None,
            subset_y=None,
            mask_sim=False,
            pad=(0, 0),
            progress=None,
        )
        self.shift_cpu = np.concatenate(self.shift_cpu, axis=1)
        self.sx = self.shift_cpu[1]
        self.sy = self.shift_cpu[0]
        # TODO: send to GPU and do both calcs there.
        self.low_e_viewer.projections.shrunken_data = shift_projections(
            self.low_e_viewer.projections.shrunken_data,
            self.sx,
            self.sy,
        )
        self.low_e_viewer.plot_shrunken()
        # self.low_e_viewer.diff_images = np.array(
        #     [x / np.mean(x) for x in self.low_e_viewer.viewer_parent.original_images]
        # ) - np.array([x / np.mean(x) for x in self.low_e_viewer.original_images])
        self.low_e_viewer.diff_on = False
        self.low_e_viewer._disable_diff_callback = True
        self.low_e_viewer.diff_button.disabled = False
        self.low_e_viewer.start_button.button_style = "success"
        self.low_e_viewer.start_button.icon = "fa-check-square"
        self.low_e_viewer._disable_diff_callback = False
        self.low_e_viewer.save_button.disabled = False


    def save_shifted_projections(self, change):

        self.make_2E_dir()
        self.metadata.filedir = self.filedir
        self.metadata.set_metadata(self)
        self.metadata.save_metadata()
        self.low_e_uploader.projections.data = da.from_array(
            self.low_e_uploader.projections.data
        )
        hist, r, bins, percentile = self.low_e_uploader.projections._dask_hist()
        grp = self.low_e_uploader.projections.hdf_key_norm + "/"
        data_dict = {
            self.low_e_uploader.projections.hdf_key_norm_proj: self.low_e_uploader.projections.data,
            grp + self.low_e_uploader.projections.hdf_key_bin_frequency: hist[0],
            grp + self.low_e_uploader.projections.hdf_key_bin_edges: hist[1],
            grp + self.low_e_uploader.projections.hdf_key_image_range: r,
            grp + self.low_e_uploader.projections.hdf_key_percentile: percentile,
        }
        self.low_e_uploader.projections.dask_data_to_h5(
            data_dict, savedir=self.filedir
        )
        self.low_e_uploader.projections._dask_bin_centers(
            grp, write=True, savedir=self.filedir
        )

    def make_2E_dir(self):
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S-2E")
        self.filedir = pathlib.Path(self.low_e_viewer.projections.filedir) / dt_string
        self.filedir.mkdir()