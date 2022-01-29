from abc import ABC, abstractmethod
from bqplot_image_gl import ImageGL
from ipywidgets import *
import bqplot as bq
import numpy as np
import copy


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

    def __init__(self, imagestack=None, title=None, dimensions=("1024px", "1024px")):
        super().__init__(imagestack, title)
        self.dimensions = dimensions
        self.current_plot_axis = 0
        self.rectangle_selector_on = False
        self.rectangle_selection = [[0, 1], [0, 1]]
        self.current_interval = 300
        # self.
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
            padding_x=0,
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
            interval=self.current_interval,
            disabled=False,
        )
        # Go faster on play button
        self.plus_button = Button(
            icon="plus",
            layout=self.button_layout,
            style=self.button_font,
        )
        # Go slower on play button
        self.minus_button = Button(
            icon="minus",
            layout=self.button_layout,
            style=self.button_font,
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
        # Rectangle selector button
        self.rectangle_selector_button = Button(
            icon="far square",
            layout=self.button_layout,
            style=self.button_font,
        )
        # Rectangle selector
        self.rectangle_selector = bq.interacts.BrushSelector(
            x_scale=self.scale_x,
            y_scale=self.scale_y,
        )
        # Histogram
        self.hist = BqImHist(self)

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

        # Swap axes button
        self.swap_axes_button.on_click(self.swap_axes)

        # Removing high/low intensities button
        self.rm_high_low_int_button.on_click(self.rm_high_low_int)

        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_pixel_range, "selected")

        # Rectangle selector button
        self.rectangle_selector_button.on_click(self.rectangle_select)

        # Faster play interval
        self.plus_button.on_click(self.speed_up)

        # Slower play interval
        self.minus_button.on_click(self.slow_down)

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
        self.current_image_ind = change.new
        self.hist.update()

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
        self.hist.update()

    # Removing high/low intensities
    def rm_high_low_int(self, change):
        self.vmin, self.vmax = np.percentile(self.imagestack, q=(0.5, 99.5))
        self.hist.selector.selected = [self.vmin, self.vmax]

    # Rectangle selector button
    def rectangle_select(self, change):
        if self.rectangle_selector_on is False:
            self.fig.interaction = self.rectangle_selector
            self.fig.interaction.color = "red"
            self.rectangle_selector_on = True
            self.rectangle_selector_button.button_style = "success"
        else:
            self.rectangle_selector_button.button_style = ""
            self.rectangle_selector_on = False
            self.fig.interaction = None

    # Rectangle selector to update projection range
    def rectangle_to_pixel_range(self, *args):
        self.pixel_range = self.rectangle_selector.selected
        self.pixel_range_x = [
            int(x) for x in np.around(self.pixel_range[:, 0] * self.pxX)
        ]
        self.pixel_range_y = np.around(self.pixel_range[:, 1] * (self.pxY - 1))
        self.pixel_range_y = [int(x) for x in self.pixel_range_y]

    # Reset
    def reset(self, *args):
        if self.current_plot_axis == 1:
            self.swap_axes()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.hist.selector.selected = None
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0

    def speed_up(self, *args):
        self.current_interval -= 50
        self.play.interval = self.current_interval

    def slow_down(self, *args):
        self.current_interval += 50
        self.play.interval = self.current_interval

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
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.hist.preflatten_imagestack(self.imagestack)

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
        center_layout = Layout(justify_content="center", align_content="center")
        header = HBox([self.scheme_dropdown], layout=header_layout)
        left_sidebar = None
        right_sidebar = None
        center = HBox([self.fig, self.hist.fig], layout=center_layout)
        self.button_box = HBox(
            [
                self.plus_button,
                self.minus_button,
                self.reset_button,
                self.rm_high_low_int_button,
                self.swap_axes_button,
                self.rectangle_selector_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox([self.button_box], layout=footer_layout)

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])


class BqImPlotter_Center(BqImPlotter_Import):
    def __init__(self):
        super().__init__()


class BqImHist:
    def __init__(self, implotter: BqImPlotter):
        self.implotter = implotter
        self.fig = bq.Figure(
            padding=0,
            fig_margin=dict(top=0, bottom=0, left=0, right=0),
        )
        self.preflatten_imagestack(self.implotter.imagestack)
        self.fig.layout.width = "100px"
        self.fig.layout.height = implotter.fig.layout.height

    def preflatten_imagestack(self, imagestack):
        self.x_sc = bq.LinearScale(
            min=float(self.implotter.vmin), max=float(self.implotter.vmax)
        )
        self.y_sc = bq.LinearScale()
        self.fig.scale_x = self.x_sc
        self.fig.scale_y = bq.LinearScale()
        self.hists = [
            bq.Bins(
                sample=image.ravel(),
                scales={
                    "x": self.x_sc,
                    "y": self.y_sc,
                },
                colors=["dodgerblue"],
                opacities=[0.75],
                orientation="horizontal",
                bins="sqrt",
                density=True,
            )
            for image in self.implotter.imagestack
        ]
        self.xvals = [x.x for x in self.hists]
        self.yvals = [x.y for x in self.hists]
        for i in range(len(self.hists)):
            ind = self.xvals[i] < self.implotter.vmin
            self.yvals[i][ind] = 0
            self.hists[i].scales["y"].max = np.max(self.yvals[i])
        self.hists_swapaxes = [
            bq.Bins(
                sample=image.ravel(),
                scales={
                    "x": self.x_sc,
                    "y": self.y_sc,
                },
                colors=["dodgerblue"],
                opacities=[0.75],
                orientation="horizontal",
                bins="sqrt",
            )
            for image in np.swapaxes(self.implotter.imagestack, 0, 1)
        ]
        self.selector = bq.interacts.BrushIntervalSelector(
            orientation="vertical", scale=self.x_sc
        )
        self.selector.observe(self.update_crange_selector, "selected")
        self.fig.marks = [self.hists[0]]
        self.fig.interaction = self.selector

    def update_crange_selector(self, *args):
        if self.selector.selected is not None:
            self.implotter.image_scale["image"].min = self.selector.selected[0]
            self.implotter.image_scale["image"].max = self.selector.selected[1]
            self.implotter.vmin = np.min(self.selector.selected[0])
            self.implotter.vmax = np.max(self.selector.selected[1])

    def update(self):
        if self.implotter.current_plot_axis == 0:
            self.hist = self.hists[self.implotter.current_image_ind]
        else:
            self.hist = self.hists_swapaxes[self.implotter.current_image_ind]
        self.fig.marks = [self.hist]


class BqImPlotter_Analysis(BqImPlotter):
    def __init__(self, plotter_parent, align_parent, dimensions=("550px", "550px")):
        super().__init__(dimensions=dimensions)
        self.pixel_range_x = (0, 10)
        self.pixel_range_y = (0, 10)
        self.align_parent = align_parent
        self.plotter_parent = plotter_parent
        self.copy_button = Button(
            icon="file-import",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.copy_button.on_click(self.copy_parent_projections)
        self.link_plotted_projections_button = Button(
            icon="unlink",
            layout=self.button_layout,
            style=self.button_font,
            disabled=True,
        )
        self.link_plotted_projections_button.on_click(self.link_plotted_projections)
        self.plots_linked = False

        self.range_from_parent_button = Button(
            icon="object-ungroup",
            layout=self.button_layout,
            style=self.button_font,
            disabled=True,
        )
        self.range_from_parent_button.on_click(self.range_from_parent)

        self.remove_data_outside_button = Button(
            description="Set data = 0 outside of current histogram range.",
            layout=Layout(width="auto"),
        )
        self.remove_data_outside_button.on_click(self.remove_data_outside)

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
        header = HBox([self.scheme_dropdown], layout=header_layout)
        left_sidebar = None
        right_sidebar = None
        center = HBox([self.fig, self.hist.fig], layout=center_layout)
        self.button_box = HBox(
            [
                self.plus_button,
                self.minus_button,
                self.reset_button,
                self.rm_high_low_int_button,
                self.swap_axes_button,
                self.copy_button,
                self.link_plotted_projections_button,
                self.range_from_parent_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [
                self.button_box,
                HBox(
                    [self.remove_data_outside_button],
                    layout=Layout(justify_content="center"),
                ),
            ]
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])

    # Image plot
    def plot(self, imagestack):
        self.imagestack = copy.copy(imagestack)
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.hist.hists = copy.copy(self.plotter_parent.hist.hists)
        self.hist.hists_swapaxes = copy.copy(self.plotter_parent.hist.hists_swapaxes)
        self.hist.selector = bq.interacts.BrushIntervalSelector(
            orientation="vertical",
            scale=bq.LinearScale(min=float(self.vmin), max=float(self.vmax)),
        )
        self.hist.selector.observe(self.hist.update_crange_selector, "selected")
        self.hist.fig.interaction = self.hist.selector
        self.hist.update()

    def copy_parent_projections(self, *args):
        self.plot(self.plotter_parent.imagestack)
        self.link_plotted_projections_button.button_style = "info"
        self.link_plotted_projections_button.disabled = False
        self.range_from_parent_button.disabled = False

    def link_plotted_projections(self, *args):
        if not self.plots_linked:
            self.plot_link = jsdlink(
                (self.plotter_parent.image_index_slider, "value"),
                (self.image_index_slider, "value"),
            )
            self.link_plotted_projections_button.button_style = "success"
            self.link_plotted_projections_button.icon = "link"
        else:
            self.plot_link.unlink()
            self.link_plotted_projections_button.button_style = "info"
            self.link_plotted_projections_button.icon = "unlink"

    def range_from_parent(self, *args):
        if (
            self.plotter_parent.rectangle_selector_button.button_style == "success"
            and self.plotter_parent.rectangle_selector.selected is not None
        ):
            imtemp = self.plotter_parent.imagestack
            lowerY = self.plotter_parent.pixel_range_y[0]
            upperY = self.plotter_parent.pixel_range_y[1]
            lowerX = self.plotter_parent.pixel_range_x[0]
            upperX = self.plotter_parent.pixel_range_x[1]
            self.imagestack = copy.deepcopy(imtemp[:, lowerY:upperY, lowerX:upperX])
            self.change_aspect_ratio()
            self.plotted_image.image = self.imagestack[
                self.plotter_parent.current_image_ind
            ]
            # This is confusing - decide on better names. The actual dimensions are
            # stored in self.projections.pixel_range_x, but this will eventually set the
            # Analysis attributes for pixel_range_x, pixel_range_y to input into
            # algorithms
            self.pixel_range_x = (lowerX, upperX)
            self.pixel_range_y = (lowerY, upperY)

    def remove_data_outside(self, *args):
        self.remove_high_indexes = self.imagestack > self.vmax
        self.imagestack[self.remove_high_indexes] = 1e-9
        self.remove_low_indexes = self.imagestack < self.vmin
        self.imagestack[self.remove_low_indexes] = 1e-9
        self.plotted_image.image = self.imagestack[0]
        self.hist.preflatten_imagestack(self.imagestack)
        self.hist.update()
