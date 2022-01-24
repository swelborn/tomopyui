from abc import ABC, abstractmethod
from bqplot_image_gl import ImageGL
from ipywidgets import *
import bqplot as bq
import numpy as np


class ImPlotterBase(ABC):
    def __init__(self, imagestack=None, title=None):
        if imagestack is None:
            self.imagestack = np.random.rand(100, 100, 100)
        else:
            self.imagestack = imagestack
        self.current_image_ind = 0
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.title = title
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.aspect_ratio = self.pxX / self.pxY
        self.fig = None

    @abstractmethod
    def _init_fig(self):
        ...

    @abstractmethod
    def _init_widgets(self):
        ...

    @abstractmethod
    def _init_observes(self):
        ...

    @abstractmethod
    def plot(self, imagestack):
        ...


class BqImPlotter(ImPlotterBase, ABC):
    lin_schemes = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "OrRd",
        "PuBu",
        "BuPu",
        "Oranges",
        "BuGn",
        "YlOrBr",
        "YlGn",
        "Reds",
        "RdPu",
        "Greens",
        "YlGnBu",
        "Purples",
        "GnBu",
        "Greys",
        "YlOrRd",
        "PuRd",
        "Blues",
        "PuBuGn",
    ]

    def __init__(self, imagestack=None, title=None, dimensions=("1024px", "181px")):
        super().__init__(imagestack, title)
        self.dimensions = dimensions
        self.current_plot_axis = 0
        self._init_fig()
        self._init_widgets()
        self._init_observes()
        self._init_links()

    def _init_fig(self):
        self.scale_x = bq.LinearScale(min=0, max=1)
        self.scale_y = bq.LinearScale(min=1, max=0)
        self.scale_x_y = {"x": self.scale_x, "y": self.scale_y}
        self.fig = bq.Figure(
            scales=self.scale_x_y,
            fig_margin=dict(top=0, bottom=0, left=0, right=0),
            padding_y=0,
        )
        self.fig.layout.fig_margin = dict(top=0, bottom=0, left=0, right=0)
        self.image_scale = {
            "x": self.scale_x,
            "y": self.scale_y,
            "image": bq.ColorScale(
                min=self.vmin,
                max=self.vmax,
                scheme="viridis",
            ),
        }
        self.plotted_image = ImageGL(
            image=self.imagestack[self.current_image_ind],
            scales=self.image_scale,
        )
        self.fig.marks = (self.plotted_image,)
        self.fig.layout.width = self.dimensions[0]
        self.fig.layout.height = self.dimensions[1]
        if self.title is not None:
            self.fig.title = f"{self.title}"

    def _init_widgets(self):

        # Styles and layouts
        self.button_font = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")

        # Image index slider
        self.image_index_slider = IntSlider(
            value=0,
            min=0,
            max=self.imagestack.shape[0] - 1,
            step=1,
        )
        # Image index play button
        self.play = Play(
            value=0,
            min=0,
            max=self.imagestack.shape[0] - 1,
            step=1,
            interval=200,
            disabled=False,
        )
        # Color range slider
        self.color_range_slider = FloatRangeSlider(
            description="Color range:",
            min=self.vmin,
            max=self.vmax,
            step=(self.vmax - self.vmin) / 1000,
            value=(self.vmin, self.vmax),
        )
        # Color scheme dropdown menu
        self.scheme_dropdown = Dropdown(
            description="Scheme:", options=self.lin_schemes, value="viridis"
        )
        # Swap axes button
        self.swap_axes_button = Button(
            icon="random",
            layout=self.button_layout,
            style=self.button_font,
        )
        # Remove high/low intensities button
        self.rm_high_low_int_button = Button(
            icon="adjust",
            layout=self.button_layout,
            style=self.button_font,
        )

        # Reset button
        self.reset_button = Button(
            icon="redo",
            layout=self.button_layout,
            style=self.button_font,
        )

    def _init_observes(self):

        # Image index slider
        self.image_index_slider.observe(self.change_image, names="value")

        # Color scheme dropdown menu
        self.scheme_dropdown.observe(self.update_scheme, names="value")

        # Color range slider
        self.color_range_slider.observe(self.update_color_range, names="value")

        # Swap axes button
        self.swap_axes_button.on_click(self.swap_axes)

        # Removing high/low intensities button
        self.rm_high_low_int_button.on_click(self.rm_high_low_int)

        # Reset button
        self.reset_button.on_click(self.reset)

    def _init_links(self):

        # Image index slider and play button
        jslink((self.play, "value"), (self.image_index_slider, "value"))
        jslink((self.play, "min"), (self.image_index_slider, "min"))
        jslink((self.play, "max"), (self.image_index_slider, "max"))

    # -- Callback functions ------------------------------------------------------------

    # Image index
    def change_image(self, change):
        self.plotted_image.image = self.imagestack[change.new]

    # Color range
    def update_color_range(self, change):
        self.image_scale["image"].min = change["new"][0]
        self.image_scale["image"].max = change["new"][1]

    # Scheme
    def update_scheme(self, *args):
        self.image_scale["image"].scheme = self.scheme_dropdown.value

    # Swap axes
    def swap_axes(self, *args):
        self.imagestack = np.swapaxes(self.imagestack, 0, 1)
        self.change_aspect_ratio()
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.plotted_image.image = self.imagestack[self.image_index_slider.value]
        if self.current_plot_axis == 0:
            self.current_plot_axis = 1
        else:
            self.current_plot_axis = 0

    # Removing high/low intensities
    def rm_high_low_int(self, change):
        self.vmin, self.vmax = np.percentile(self.imagestack, q=(0.5, 99.5))
        self.color_range_slider.max = self.vmax
        self.color_range_slider.min = self.vmin
        self.color_range_slider.value = (self.vmin, self.vmax)

    # Reset
    def reset(self, *args):
        if self.current_plot_axis == 1:
            self.swap_axes()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.color_range_slider.min = self.vmin
        self.color_range_slider.max = self.vmax
        self.color_range_slider.value = (self.vmin, self.vmax)

    # Image plot
    def plot(self, imagestack):
        self.imagestack = imagestack
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.color_range_slider.min = self.vmin
        self.color_range_slider.max = self.vmax
        self.color_range_slider.value = (self.vmin, self.vmax)

    # -- Other methods -----------------------------------------------------------------

    def change_aspect_ratio(self):
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.aspect_ratio = self.pxX / self.pxY
        if self.aspect_ratio > self.fig.max_aspect_ratio:
            self.fig.max_aspect_ratio = self.aspect_ratio
            self.fig.min_aspect_ratio = self.aspect_ratio
        else:
            self.fig.min_aspect_ratio = self.aspect_ratio
            self.fig.max_aspect_ratio = self.aspect_ratio
        self.fig.layout.height = str(int(550 / self.aspect_ratio)) + "px"

    @abstractmethod
    def create_app(self):
        ...


class BqImPlotter_Import(BqImPlotter):
    def __init__(self, dimensions=("550px", "550px")):
        super().__init__(dimensions=dimensions)

    def create_app(self):

        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        header_layout = Layout(justify_content="center")
        footer_layout = Layout(justify_content="center")
        center_layout = Layout(justify_content="center")
        header = HBox(
            [self.scheme_dropdown, self.color_range_slider], layout=header_layout
        )
        left_sidebar = None
        right_sidebar = None
        center = HBox([self.fig], layout=center_layout)
        self.button_box = HBox(
            [
                self.reset_button,
                self.rm_high_low_int_button,
                self.swap_axes_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox([self.button_box], layout=footer_layout)

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])
