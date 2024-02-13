from ipywidgets import *

from tomopyui._sharedvars import *
from tomopyui.widgets.view import (
    BqImViewer_Projections_Parent,
    BqImViewer_TwoEnergy_Low,
    BqImViewerBase,
)


class BqImViewer_HDF5(BqImViewerBase):
    def __init__(self):

        super().__init__()
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector."
        )
        self.from_hdf = True
        self.from_npy = False

    def create_app(self):
        self.button_box = HBox(
            self.init_buttons,
            layout=self.footer_layout,
        )

        footer2 = VBox(
            [
                self.button_box,
                HBox(
                    [
                        self.status_bar_xrange,
                        self.status_bar_yrange,
                        self.status_bar_intensity,
                    ],
                    layout=self.footer_layout,
                ),
                HBox(
                    [
                        self.status_bar_xdistance,
                        self.status_bar_ydistance,
                    ],
                    layout=self.footer_layout,
                ),
            ],
            layout=self.footer_layout,
        )
        footer = VBox([self.footer1, footer2])
        self.app = VBox([self.header, self.center, footer])

    def plot(self, projections, hdf_handler):
        self.projections = projections
        self.hdf_handler = hdf_handler
        self.filedir = self.projections.filedir
        self.px_size = self.projections.px_size
        self.hist.precomputed_hist = self.projections.hist
        self.original_images = self.projections.data
        if self.hdf_handler.loaded_ds:
            self.images = self.projections.data_ds
        else:
            self.images = self.projections.data
        self.set_state_on_plot()

    # Downsample the plot view
    def downsample_viewer(self, *args):
        if self.hdf_handler.turn_off_callbacks:
            return
        self.ds_factor = self.ds_dropdown.value
        if self.ds_factor != -1:
            self.hdf_handler.load_ds(str(self.ds_factor))
            self.original_images = self.projections.data
            self.images = self.projections.data_ds
            self.hist.precomputed_hist = self.projections.hist
        elif self.ds_factor == -1:
            self.hdf_handler.load_any()
            self.original_images = self.projections.data
            self.images = self.projections.data
            self.hist.precomputed_hist = self.projections.hist

        self.plotted_image.image = self.images[self.image_index_slider.value]
        self.change_aspect_ratio()


class BqImViewer_HDF5_Align_To(BqImViewer_HDF5):
    def __init__(self):
        super().__init__()
        # Rectangle selector button
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector. Select a region here "
            "to do phase correlation on."
        )
        self.rectangle_selector_on = False
        self.rectangle_select(None)
        self.viewing = False

    # Rectangle selector to update projection range
    def rectangle_to_px_range(self, *args):
        BqImViewer_Projections_Parent.rectangle_to_px_range(self)
        self.viewer_child.match_rectangle_selector_range_parent()


class BqImViewer_HDF5_Align(BqImViewer_TwoEnergy_Low):
    def __init__(self, viewer_parent):
        super().__init__(viewer_parent)
        self.from_hdf = True
        self.ds_dropdown = self.viewer_parent.ds_dropdown
        self.scheme_dropdown = self.viewer_parent.scheme_dropdown
        self.start_button.disabled = False
        self.save_button.disabled = False

    def make_buttons(self):
        self.init_buttons = [
            self.plus_button,
            self.minus_button,
            self.reset_button,
            self.rm_high_low_int_button,
            self.rectangle_selector_button,
        ]
        self.all_buttons = self.init_buttons
        super().make_buttons()
        del self.scale_button
        self.diff_button.on_click(self.switch_to_diff)

    def add_buttons(self):
        self.all_buttons.insert(-1, self.diff_button)
        self.all_buttons.insert(-1, self.link_plotted_projections_button)
        self.all_buttons.insert(-1, self.start_button)
        self.all_buttons.insert(-1, self.save_button)

    def plot(self, projections, hdf_handler):
        BqImViewer_HDF5.plot(self, projections, hdf_handler)

    def downsample_viewer(self, *args):
        pass
