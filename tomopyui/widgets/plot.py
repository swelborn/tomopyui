from abc import ABC, abstractmethod
from bqplot_image_gl import ImageGL
from ipywidgets import *
import bqplot as bq
import numpy as np
import copy
from skimage.transform import rescale  # look for better option
from tomopyui._sharedvars import *
import pathlib
from bqplot_image_gl.interacts import MouseInteraction, keyboard_events, mouse_events
from bqplot import PanZoom

# for movie saving
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ImPlotterBase(ABC):
    def __init__(self, imagestack=None, title=None):
        if imagestack is None:
            self.original_imagestack = np.random.rand(100, 100, 100)
            self.imagestack = self.original_imagestack
        else:
            self.original_imagestack = imagestack
            self.imagestack = imagestack
        self.current_image_ind = 0
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.title = title
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.aspect_ratio = self.pxX / self.pxY
        self.fig = None
        self.downsample_factor = 0.2
        self.precomputed_hists = None

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
        self.panzoom = PanZoom(scales={"x": [self.scale_x], "y": [self.scale_y]})
        self.msg_interaction = MouseInteraction(
            x_scale=self.scale_x,
            y_scale=self.scale_y,
            move_throttle=70,
            next=self.panzoom,
            events=keyboard_events + mouse_events,
        )
        self.fig.interaction = self.msg_interaction

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
            tooltip="Speed up play slider.",
        )

        # Go slower on play button
        self.minus_button = Button(
            icon="minus",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Slow down play slider slider.",
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
            tooltip="Swap axes",
        )

        # Remove high/low intensities button
        self.rm_high_low_int_button = Button(
            icon="adjust",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Remove high and low intensities from view.",
        )

        # Save movie button
        self.save_movie_button = Button(
            icon="file-video",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Save a movie of these images.",
        )

        # Histogram
        self.hist = BqImHist(self)

        # Downsample (for viewing) textbox
        self.downsample_viewer_textbox = FloatText(
            description="Viewer sampling:",
            min=0.1,
            max=1,
            value=self.downsample_factor,
            step=0.1,
            style=extend_description_style,
        )

        # Downsample (for viewing) button
        self.downsample_viewer_button = Button(
            icon="arrow-down",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Downsample for faster viewing. Does not impact analysis data.",
        )

        # Reset button
        self.reset_button = Button(
            icon="redo",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Reset to original view.",
        )

        # Intensity status bar
        self.status_bar_intensity = Label()

    def _init_observes(self):

        # Image index slider
        self.image_index_slider.observe(self.change_image, names="value")

        # Color scheme dropdown menu
        self.scheme_dropdown.observe(self.update_scheme, names="value")

        # Swap axes button
        self.swap_axes_button.on_click(self.swap_axes)

        # Removing high/low intensities button
        self.rm_high_low_int_button.on_click(self.rm_high_low_int)

        # Faster play interval
        self.plus_button.on_click(self.speed_up)

        # Slower play interval
        self.minus_button.on_click(self.slow_down)

        # Save a movie at the current vmin/vmax
        self.save_movie_button.on_click(self.save_movie)

        # Downsample (for viewing) button
        self.downsample_viewer_textbox.observe(self.change_downsample_button, "value")

        # Downsample (for viewing) button
        self.downsample_viewer_button.on_click(self.downsample_viewer)

        # Reset button
        self.reset_button.on_click(self.reset)

        # Zoom/intensity
        self.msg_interaction.on_msg(self.on_mouse_msg_intensity)

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
        # self.hist.update()

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
        # self.hist.update()

    # Removing high/low intensities
    def rm_high_low_int(self, change):
        self.vmin, self.vmax = np.percentile(self.imagestack, q=(0.5, 99.5))
        self.hist.selector.selected = [self.vmin, self.vmax]

    # Change downsample to upsample
    def change_downsample_button(self, *args):
        if self.downsample_viewer_textbox.value > self.downsample_factor:
            self.downsample_viewer_button.disabled = False
            self.downsample_viewer_button.icon = "arrow-up"
            self.downsample_viewer_button.button_style = "success"
        elif self.downsample_viewer_textbox.value < self.downsample_factor:
            self.downsample_viewer_button.disabled = False
            self.downsample_viewer_button.icon = "arrow-down"
            self.downsample_viewer_button.button_style = "success"
        else:
            self.downsample_viewer_button.icon = "sort"
            self.downsample_viewer_button.disabled = True
            self.downsample_viewer_button.button_style = ""

    # Downsample the plot view
    def downsample_viewer(self, *args):
        self.downsample_factor = self.downsample_viewer_textbox.value
        self.downsample_imagestack(self.original_imagestack)
        self.change_downsample_button()

    def downsample_imagestack(self, imagestack):
        self.imagestack = copy.deepcopy(imagestack)
        self.imagestack = rescale(
            self.imagestack,
            (1, self.downsample_factor, self.downsample_factor),
            anti_aliasing=False,
        )
        self.image_index_slider.value = 0
        self.plotted_image.image = self.imagestack[0]

    # Reset
    def reset(self, *args):
        if self.current_plot_axis == 1:
            self.swap_axes()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_scale["image"].min = self.vmin
        self.image_scale["image"].max = self.vmax
        self.hist.selector.selected = None
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.rm_high_low_int(None)

    # Speed up playback of play button
    def speed_up(self, *args):
        self.current_interval -= 50
        self.play.interval = self.current_interval

    # Slow down playback of play button
    def slow_down(self, *args):
        self.current_interval += 50
        self.play.interval = self.current_interval

    # Save movie
    def save_movie(self, *args):
        self.save_movie_button.icon = "fas fa-cog fa-spin fa-lg"
        self.save_movie_button.button_style = "info"
        fig, ax = plt.subplots(figsize=(10, 5))
        _ = ax.set_axis_off()
        _ = fig.patch.set_facecolor("black")
        ims = []
        vmin = self.image_scale["image"].min
        vmax = self.image_scale["image"].max
        for image in self.original_imagestack:
            im = ax.imshow(image, animated=True, vmin=vmin, vmax=vmax)
            ims.append([im])
        ani = animation.ArtistAnimation(
            fig, ims, interval=300, blit=True, repeat_delay=1000
        )
        writer = animation.FFMpegWriter(
            fps=20, codec=None, bitrate=1000, extra_args=None, metadata=None
        )
        _ = ani.save(str(pathlib.Path(self.filedir / "movie.mp4")), writer=writer)
        self.save_movie_button.icon = "file-video"
        self.save_movie_button.button_style = "success"

    # Image plot
    def plot(
        self, imagestack, filedir, io=None, precomputed_hists=None, nm_per_px=None
    ):
        self.precomputed_hists = precomputed_hists
        self.io = io
        self.filedir = filedir
        self.nm_per_px = nm_per_px
        self.pxX = imagestack.shape[2]
        self.pxY = imagestack.shape[1]
        self.original_imagestack = imagestack
        if self.io is not None:
            self.imagestack = np.array(self.io.data_ds[1])
            self.downsample_factor = 0.25
            self.downsample_viewer_textbox.value = self.downsample_factor
        else:
            self.downsample_imagestack(imagestack)
        self.change_downsample_button()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.hist.preflatten_imagestack(self.imagestack)
        self.rm_high_low_int(None)

    # Intensity message
    def on_mouse_msg_intensity(self, interaction, data, buffers):
        # it might be a good idea to throttle on the Python side as well, for instance when many computations
        # happen, we can effectively ignore the queue of messages
        if data["event"] == "mousemove":
            domain_x = data["domain"]["x"]
            domain_y = data["domain"]["y"]
            normalized_x = (domain_x - self.plotted_image.x[0]) / (
                self.plotted_image.x[1] - self.plotted_image.x[0]
            )
            normalized_y = (domain_y - self.plotted_image.y[0]) / (
                self.plotted_image.y[1] - self.plotted_image.y[0]
            )
            # TODO: think about +/-1 and pixel edges
            pixel_x = int(np.floor(normalized_x * self.plotted_image.image.shape[1]))
            pixel_y = int(np.floor(normalized_y * self.plotted_image.image.shape[0]))
            if (
                pixel_x >= 0
                and pixel_x < self.plotted_image.image.shape[1]
                and pixel_y >= 0
                and pixel_y < self.plotted_image.image.shape[0]
            ):
                value = str(round(self.plotted_image.image[pixel_y, pixel_x], 5))
            else:
                value = "Out of range"
            msg = f"Intensity={value}"
            self.status_bar_intensity.value = msg
        elif data["event"] == "mouseleave":
            self.status_bar_intensity.value = ""
        elif data["event"] == "mouseenter":
            self.status_bar_intensity.value = "Almost there..."  # this is is not visible because mousemove overwrites the msg
        else:
            self.status_bar_intensity.value = f"click {data}"

    # -- Other methods -----------------------------------------------------------------
    def change_aspect_ratio(self):
        pxX = self.imagestack.shape[2]
        pxY = self.imagestack.shape[1]
        self.aspect_ratio = pxX / pxY
        if self.aspect_ratio >= 1:
            if self.aspect_ratio > self.fig.max_aspect_ratio:
                self.fig.max_aspect_ratio = self.aspect_ratio
                self.fig.min_aspect_ratio = self.aspect_ratio
            else:
                self.fig.min_aspect_ratio = self.aspect_ratio
                self.fig.max_aspect_ratio = self.aspect_ratio
            # This is set to default dimensions of 550, not great:
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
        header_layout = Layout(justify_content="center", align_items="center")
        footer_layout = Layout(justify_content="center", align_items="center")
        center_layout = Layout(justify_content="center", align_content="center")
        header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=header_layout,
        )
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
                self.save_movie_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [self.button_box, self.status_bar_intensity],
            layout=footer_layout,
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])


class BqImPlotter_Import_Analysis(BqImPlotter):
    def __init__(self, Analysis, dimensions=("550px", "550px")):
        self.Analysis = Analysis
        super().__init__(dimensions=dimensions)
        # Rectangle selector button
        self.rectangle_selector_button = Button(
            icon="far square",
            layout=self.button_layout,
            style=self.button_font,
            tooltip=(
                "Turn on the rectangular region selector. Select a region "
                "and copy it over to  Altered Projections."
            ),
        )
        # Rectangle selector
        self.rectangle_selector = bq.interacts.BrushSelector(
            x_scale=self.scale_x,
            y_scale=self.scale_y,
        )
        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_pixel_range, "selected")

        # Rectangle selector button
        self.rectangle_selector_button.on_click(self.rectangle_select)

        # Status bar updates
        self.status_bar_xrange = Label()
        self.status_bar_yrange = Label()
        self.status_bar_xrange.value = ""
        self.status_bar_yrange.value = ""
        self.status_bar_xdistance = Label()
        self.status_bar_ydistance = Label()
        self.status_bar_xdistance.value = ""
        self.status_bar_ydistance.value = ""

    def create_app(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        header_layout = Layout(justify_content="center", align_items="center")
        footer_layout = Layout(justify_content="center")
        center_layout = Layout(justify_content="center", align_content="center")
        header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=header_layout,
        )
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
                self.save_movie_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [
                self.button_box,
                HBox(
                    [
                        self.status_bar_xrange,
                        self.status_bar_yrange,
                        self.status_bar_intensity,
                    ],
                    layout=footer_layout,
                ),
                HBox(
                    [
                        self.status_bar_xdistance,
                        self.status_bar_ydistance,
                    ],
                    layout=footer_layout,
                ),
            ],
            layout=footer_layout,
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])

    def plot(self):
        self.original_imagestack = self.Analysis.Import.projections.data
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = np.array(self.Analysis.Import.projections.data_ds[1])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.nm_per_px = self.Analysis.Import.projections.nm_per_px
        self.precomputed_hists = self.Analysis.Import.projections.hists
        # self.downsample_imagestack(self.original_imagestack)
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_scale["image"].min = float(self.vmin)
        self.image_scale["image"].max = float(self.vmax)
        self.pixel_range_x = [0, self.pxX - 1]
        self.pixel_range_y = [0, self.pxY - 1]
        self.pixel_range = [self.pixel_range_x, self.pixel_range_y]
        # self.update_pixel_range_status_bar()
        self.hist.selector.selected = None
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.hist.preflatten_imagestack(self.imagestack)
        self.rm_high_low_int(None)
        self.change_downsample_button()

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
            self.status_bar_xrange.value = ""
            self.status_bar_yrange.value = ""
            self.status_bar_xdistance.value = ""
            self.status_bar_ydistance.value = ""
            self.fig.interaction = self.msg_interaction

    # Rectangle selector to update projection range
    def rectangle_to_pixel_range(self, *args):
        self.pixel_range = self.rectangle_selector.selected
        self.pixel_range = np.where(self.pixel_range < 0, 0, self.pixel_range)
        self.pixel_range = np.where(self.pixel_range > 1, 1, self.pixel_range)
        self.pixel_range_x = [
            int(x) for x in np.around(self.pixel_range[:, 0] * (self.pxX - 1))
        ]
        self.pixel_range_y = np.around(self.pixel_range[:, 1] * (self.pxY - 1))
        self.pixel_range_y = [int(x) for x in self.pixel_range_y]
        if self.nm_per_px is not None:
            self.nm_x = int(
                (self.pixel_range_x[1] - self.pixel_range_x[0]) * self.nm_per_px
            )
            self.nm_y = int(
                (self.pixel_range_y[1] - self.pixel_range_y[0]) * self.nm_per_px
            )
            self.micron_x = round(self.nm_x / 1000, 2)
            self.micron_y = round(self.nm_y / 1000, 2)
        self.update_pixel_range_status_bar()

    def update_pixel_range_status_bar(self):
        self.status_bar_xrange.value = f"X Pixel Range: {self.pixel_range_x} | "
        self.status_bar_yrange.value = f"Y Pixel Range: {self.pixel_range_y}"
        if self.nm_x < 1000:
            self.status_bar_xdistance.value = f"X Distance (nm): {self.nm_x} | "
        else:
            self.status_bar_xdistance.value = f"X Distance (μm): {self.micron_x} | "
        if self.nm_y < 1000:
            self.status_bar_ydistance.value = f"Y Distance (nm): {self.nm_y}"
        else:
            self.status_bar_ydistance.value = f"Y Distance (μm): {self.micron_y}"


class BqImPlotter_Center(BqImPlotter_Import_Analysis):
    def __init__(self, Center):
        super().__init__(None)
        self.Center = Center
        self.center_line_on = False
        self.center_line = bq.Lines(
            x=[0.5, 0.5],
            y=[0, 1],
            colors="red",
            stroke_width=3,
            scales={"x": self.scale_x, "y": self.scale_y},
        )
        self.center_line_button = Button(
            icon="align-center",
            layout=self.button_layout,
            style=self.button_font,
            tooltip=("Turn on center of rotation line"),
        )
        self.center_line_button.on_click(self.center_line_on_update)

    def create_app(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        header_layout = Layout(justify_content="center", align_items="center")
        footer_layout = Layout(justify_content="center")
        center_layout = Layout(justify_content="center", align_content="center")
        header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=header_layout,
        )
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
                self.center_line_button,
                self.save_movie_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [
                self.button_box,
                HBox(
                    [
                        self.status_bar_xrange,
                        self.status_bar_yrange,
                        self.status_bar_intensity,
                    ],
                    layout=footer_layout,
                ),
                HBox(
                    [
                        self.status_bar_xdistance,
                        self.status_bar_ydistance,
                    ],
                    layout=footer_layout,
                ),
            ],
            layout=footer_layout,
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])

    # Rectangle selector button
    def center_line_on_update(self, *args):
        if self.center_line_on is False:
            self.center_line_on = True
            self.center_line_button.button_style = "success"
            self.fig.marks = (
                self.plotted_image,
                self.center_line,
            )
        else:
            self.center_line_on = False
            self.center_line_button.button_style = ""
            self.fig.marks = (self.plotted_image,)

    def plot(self):
        self.original_imagestack = self.Center.Import.projections.data
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = np.array(self.Center.Import.projections.data_ds[1])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.nm_per_px = self.Center.Import.projections.nm_per_px
        self.precomputed_hists = self.Center.Import.projections.hists
        # self.downsample_imagestack(self.original_imagestack)
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_scale["image"].min = float(self.vmin)
        self.image_scale["image"].max = float(self.vmax)
        self.pixel_range_x = [0, self.pxX - 1]
        self.pixel_range_y = [0, self.pxY - 1]
        self.pixel_range = [self.pixel_range_x, self.pixel_range_y]
        # self.update_pixel_range_status_bar()
        self.hist.selector.selected = None
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.hist.preflatten_imagestack(self.imagestack)
        self.rm_high_low_int(None)
        self.change_downsample_button()
        self.center_line_on = False
        self.center_line_on_update()


class BqImPlotter_Center_Recon(BqImPlotter_Import):
    def __init__(self):
        super().__init__()


class BqImPlotter_Altered_Analysis(BqImPlotter_Import_Analysis):
    def __init__(self, plotter_parent, Analysis, dimensions=("550px", "550px")):
        self.Analysis = Analysis
        super().__init__(Analysis, dimensions=dimensions)
        self.plotter_parent = plotter_parent

        # Copy from plotter
        self.copy_button = Button(
            icon="file-import",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Copy data from 'Imported Projections'.",
        )
        self.copy_button.on_click(self.copy_parent_projections)

        self.link_plotted_projections_button = Button(
            icon="unlink",
            layout=self.button_layout,
            style=self.button_font,
            disabled=True,
            tooltip="Link the sliders together.",
        )
        self.link_plotted_projections_button.on_click(self.link_plotted_projections)
        self.plots_linked = False

        self.range_from_parent_button = Button(
            icon="object-ungroup",
            layout=self.button_layout,
            style=self.button_font,
            disabled=True,
            tooltip="Get range from 'Imported Projections'.",
        )
        self.range_from_parent_button.on_click(self.range_from_parent)

        self.remove_data_outside_button = Button(
            description="Set data = 0 outside of current histogram range.",
            layout=Layout(width="auto"),
        )
        self.remove_data_outside_button.on_click(self.remove_data_outside)

        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_pixel_range, "selected")

    def create_app(self):

        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        header_layout = Layout(justify_content="center", align_items="center")
        footer_layout = Layout(justify_content="center")
        center_layout = Layout(justify_content="center")
        header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=header_layout,
        )
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
                self.rectangle_selector_button,
                self.save_movie_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [
                self.button_box,
                HBox(
                    [
                        self.status_bar_xrange,
                        self.status_bar_yrange,
                        self.status_bar_intensity,
                    ],
                    layout=Layout(justify_content="center"),
                ),
                HBox(
                    [self.remove_data_outside_button],
                    layout=Layout(justify_content="center"),
                ),
            ]
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])

    # Image plot
    def plot(self):
        self.original_imagestack = copy.copy(self.plotter_parent.original_imagestack)
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = copy.copy(self.plotter_parent.imagestack)
        self.pixel_range_x = [0, self.pxX - 1]
        self.pixel_range_y = [0, self.pxY - 1]
        self.pixel_range = [self.pixel_range_x, self.pixel_range_y]
        self.subset_pixel_range_x = self.pixel_range_x
        self.subset_pixel_range_y = self.pixel_range_y
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = self.plotter_parent.vmin
        self.vmax = self.plotter_parent.vmax
        self.image_scale["image"].min = self.vmin
        self.image_scale["image"].max = self.vmax
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0

    def copy_parent_projections(self, *args):
        self.plot()
        self.hist.hists = copy.copy(self.plotter_parent.hist.hists)
        # self.hist.hists_swapaxes = copy.copy(self.plotter_parent.hist.hists_swapaxes)
        self.hist.selector = bq.interacts.BrushIntervalSelector(
            orientation="vertical",
            scale=self.plotter_parent.hist.x_sc,
        )
        self.hist.selector.observe(self.hist.update_crange_selector, "selected")
        self.hist.fig.interaction = self.hist.selector
        self.hist.fig.marks = [self.hist.hists[0]]
        self.link_plotted_projections_button.button_style = "info"
        self.link_plotted_projections_button.disabled = False
        self.range_from_parent_button.disabled = False

    def link_plotted_projections(self, *args):
        if not self.plots_linked:
            self.plots_linked = True
            self.plot_link = jsdlink(
                (self.plotter_parent.image_index_slider, "value"),
                (self.image_index_slider, "value"),
            )
            self.link_plotted_projections_button.button_style = "success"
            self.link_plotted_projections_button.icon = "link"
        else:
            self.plots_linked = False
            self.plot_link.unlink()
            self.link_plotted_projections_button.button_style = "info"
            self.link_plotted_projections_button.icon = "unlink"

    def range_from_parent(self, *args):
        if (
            self.plotter_parent.rectangle_selector_button.button_style == "success"
            and self.plotter_parent.rectangle_selector.selected is not None
        ):
            imtemp = self.plotter_parent.imagestack
            lowerY = int(
                self.plotter_parent.pixel_range_y[0]
                * self.plotter_parent.downsample_factor
            )
            upperY = int(
                self.plotter_parent.pixel_range_y[1]
                * self.plotter_parent.downsample_factor
            )
            lowerX = int(
                self.plotter_parent.pixel_range_x[0]
                * self.plotter_parent.downsample_factor
            )
            upperX = int(
                self.plotter_parent.pixel_range_x[1]
                * self.plotter_parent.downsample_factor
            )
            self.imagestack = copy.deepcopy(imtemp[:, lowerY:upperY, lowerX:upperX])
            self.change_aspect_ratio()
            self.plotted_image.image = self.imagestack[
                self.plotter_parent.current_image_ind
            ]
            # This is confusing - decide on better names. The actual dimensions are
            # stored in self.projections.pixel_range_x, but this will eventually set the
            # Analysis attributes for pixel_range_x, pixel_range_y to input into
            # algorithms
            self.pixel_range_x = (
                self.plotter_parent.pixel_range_x[0],
                self.plotter_parent.pixel_range_x[1],
            )
            self.pixel_range_y = (
                self.plotter_parent.pixel_range_y[0],
                self.plotter_parent.pixel_range_y[1],
            )

    # Rectangle selector to update projection range
    def rectangle_to_pixel_range(self, *args):
        self.pixel_range = self.rectangle_selector.selected
        x_len = self.pixel_range_x[1] - self.pixel_range_x[0]
        y_len = self.pixel_range_y[1] - self.pixel_range_y[0]
        lowerX = int(self.pixel_range[0, 0] * x_len + self.pixel_range_x[0])
        upperX = int(self.pixel_range[1, 0] * x_len + self.pixel_range_x[0])
        lowerY = int(self.pixel_range[0, 1] * y_len + self.pixel_range_y[0])
        upperY = int(self.pixel_range[1, 1] * y_len + self.pixel_range_y[0])
        self.printed_range_x = [lowerX, upperX]
        self.printed_range_y = [lowerY, upperY]
        self.Analysis.subset_range_x = [
            x - self.pixel_range_x[0] for x in self.printed_range_x
        ]
        self.Analysis.subset_range_y = [
            y - self.pixel_range_y[0] for y in self.printed_range_y
        ]
        self.update_pixel_range_status_bar()

    # Rectangle selector button
    def rectangle_select(self, change):
        if self.rectangle_selector_on is False:
            self.fig.interaction = self.rectangle_selector
            self.fig.interaction.color = "magenta"
            self.rectangle_selector_on = True
            self.rectangle_selector_button.button_style = "success"
            self.Analysis.use_subset_correlation_checkbox.value = True
        else:
            self.rectangle_selector_button.button_style = ""
            self.rectangle_selector_on = False
            self.status_bar_xrange.value = ""
            self.status_bar_yrange.value = ""
            self.fig.interaction = self.msg_interaction
            self.Analysis.use_subset_correlation_checkbox.value = False

    def update_pixel_range_status_bar(self):
        self.status_bar_xrange.value = (
            f"Phase Correlation X Pixel Range: {self.printed_range_x} | "
        )
        self.status_bar_yrange.value = (
            f"Phase Correlation Y Pixel Range: {self.printed_range_y}"
        )

    def remove_data_outside(self, *args):
        self.remove_high_indexes = self.original_imagestack > self.vmax
        self.original_imagestack[self.remove_high_indexes] = 1e-9
        self.remove_low_indexes = self.original_imagestack < self.vmin
        self.original_imagestack[self.remove_low_indexes] = 1e-9
        self.plotted_image.image = self.original_imagestack[0]
        self.remove_high_indexes = self.imagestack > self.vmax
        self.imagestack[self.remove_high_indexes] = 1e-9
        self.remove_low_indexes = self.imagestack < self.vmin
        self.imagestack[self.remove_low_indexes] = 1e-9
        self.plotted_image.image = self.imagestack[0]
        self.hist.preflatten_imagestack(self.imagestack)


class BqImPlotter_DataExplorer(BqImPlotter):
    def __init__(self, plotter_parent=None, dimensions=("550px", "550px")):
        super().__init__(dimensions=dimensions)
        self.plotter_parent = plotter_parent
        self.link_plotted_projections_button = Button(
            icon="unlink",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.link_plotted_projections_button.on_click(self.link_plotted_projections)
        self.plots_linked = False
        self.downsample_factor = 0.5

    def create_app(self):

        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        header_layout = Layout(justify_content="center", align_items="center")
        footer_layout = Layout(justify_content="center")
        center_layout = Layout(justify_content="center")
        header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=header_layout,
        )
        left_sidebar = None
        right_sidebar = None
        center = HBox([self.fig, self.hist.fig], layout=center_layout)
        self.button_box = HBox(
            [
                self.plus_button,
                self.minus_button,
                self.reset_button,
                self.rm_high_low_int_button,
                self.link_plotted_projections_button,
                self.swap_axes_button,
                self.save_movie_button,
            ],
            layout=footer_layout,
        )
        footer1 = HBox([self.play, self.image_index_slider], layout=footer_layout)
        footer2 = VBox(
            [
                self.button_box,
            ]
        )

        footer = VBox([footer1, footer2])

        self.app = VBox([header, center, footer])

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

    # Save movie that will compare before/after
    def save_movie(self, *args):
        self.save_movie_button.icon = "fas fa-cog fa-spin fa-lg"
        self.save_movie_button.button_style = "info"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        _ = ax1.set_axis_off()
        _ = ax2.set_axis_off()
        _ = fig.patch.set_facecolor("black")
        ims = []
        vmin_parent = self.plotter_parent.image_scale["image"].min
        vmax_parent = self.plotter_parent.image_scale["image"].max
        vmin = self.image_scale["image"].min
        vmax = self.image_scale["image"].max
        for i in range(len(self.original_imagestack)):
            im1 = ax1.imshow(
                self.plotter_parent.original_imagestack[i],
                animated=True,
                vmin=vmin_parent,
                vmax=vmax_parent,
            )
            im2 = ax2.imshow(
                self.original_imagestack[i], animated=True, vmin=vmin, vmax=vmax
            )
            ims.append([im1, im2])
        ani = animation.ArtistAnimation(
            fig, ims, interval=300, blit=True, repeat_delay=1000
        )
        writer = animation.FFMpegWriter(
            fps=20, codec=None, bitrate=1000, extra_args=None, metadata=None
        )
        _ = ani.save(str(pathlib.Path(self.filedir / "movie.mp4")), writer=writer)
        self.save_movie_button.icon = "file-video"
        self.save_movie_button.button_style = "success"


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

        if self.implotter.precomputed_hists is not None:
            self.bin_centers = self.implotter.precomputed_hists[3]["bin_centers"]
            self.frequency = self.implotter.precomputed_hists[3]["frequency"]
            self.hists = [
                bq.Bars(
                    x=self.bin_centers,
                    y=self.frequency,
                    scales={
                        "x": self.x_sc,
                        "y": self.y_sc,
                    },
                    colors=["dodgerblue"],
                    opacities=[0.75],
                    orientation="horizontal",
                )
            ]
            ind = self.bin_centers < self.implotter.vmin
            self.frequency[ind] = 0
            self.hists[0].scales["y"].max = float(np.max(self.frequency))
        else:
            self.hists = [
                bq.Bins(
                    sample=self.implotter.imagestack.ravel(),
                    scales={
                        "x": self.x_sc,
                        "y": self.y_sc,
                    },
                    colors=["dodgerblue"],
                    opacities=[0.75],
                    orientation="horizontal",
                    bins=100,
                    density=True,
                )
            ]
            self.bin_centers = [x.x for x in self.hists]
            self.frequency = [x.y for x in self.hists]
            for i in range(len(self.hists)):
                ind = self.bin_centers[i] < self.implotter.vmin
                self.frequency[i][ind] = 0
                self.hists[i].scales["y"].max = np.max(self.frequency[i])
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
            self.implotter.vmin = self.selector.selected[0]
            self.implotter.vmax = self.selector.selected[1]

    # def update(self):
    #     if self.implotter.current_plot_axis == 0:
    #         self.hist = self.hists
    #     else:
    #         self.hist = self.hists
    # self.fig.marks = [self.hist]


class ScaleBar:
    def __init__(self):
        pass

        # attempt to make scale bar. try again later 02/05/2022
        # import bqplot as bq

        # sc_x = LinearScale(min=0, max=1)
        # sc_y = LinearScale(min=1, max=0)

        # pxX = 1024
        # nm_per_px = 30
        # px_per_micron = 1000 / nm_per_px
        # px_per_micron_half = px_per_micron / 2
        # x_px_center = 0.85 * pxX
        # num_microns = 5
        # x_coord_1 = x_px_center - px_per_micron_half * num_microns
        # x_coord_2 = x_px_center + px_per_micron_half * num_microns

        # x_line_px = np.array(
        #     [x_coord_1, x_coord_2, x_coord_2, x_coord_1], dtype=np.float32
        # )
        # x_line_px_scaled = x_line_px / pxX
        # x_line = [[0.65, 0.7, 0.7, 0.65]]
        # y_line = np.array([0.85, 0.85, 0.9, 0.9])

        # patch = Lines(
        #     x=x_line_px_scaled,
        #     y=y_line,
        #     fill_colors=["white"],
        #     fill="inside",
        #     stroke_width=0,
        #     close_path=True,
        #     scales={"x": sc_x, "y": sc_y},
        # )

        # label_text = [f"{num_microns} μm"]
        # label_pos_x = (x_line_px_scaled[0] + x_line_px_scaled[1]) / 2 - 0.08
        # label_pos_y = y_line[0] - 0.07
        # test_label = bq.Label(
        #     x=[label_pos_x],
        #     y=[label_pos_y],
        #     scales={"x": sc_x, "y": sc_y},
        #     text=label_text,
        #     default_size=30,
        #     font_weight="bolder",
        #     colors="white",
        #     update_on_move=True,
        # )

        # dimensions = ("550px", "550px")

        # fig = Figure(
        #     marks=[altered_plotter.plotted_image, patch, test_label],
        #     animation_duration=1000,
        # )
        # fig.layout.height = dimensions[1]
        # fig.layout.width = dimensions[0]
        # fig
