import bqplot as bq
import numpy as np
import copy
import pathlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from abc import ABC, abstractmethod
from bqplot_image_gl import ImageGL
from ipywidgets import *
from skimage.transform import rescale  # look for better option
from tomopyui._sharedvars import *
from bqplot_image_gl.interacts import MouseInteraction, keyboard_events, mouse_events
from bqplot import PanZoom


class BqImViewerBase(ABC):
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

    def __init__(self, imagestack=None):
        if imagestack is None:
            self.original_imagestack = np.random.rand(5, 50, 50)
            self.imagestack = self.original_imagestack
        else:
            self.original_imagestack = imagestack
            self.imagestack = imagestack
        self.current_image_ind = 0
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.pxX = self.imagestack.shape[2]
        self.pxY = self.imagestack.shape[1]
        self.aspect_ratio = self.pxX / self.pxY
        self.fig = None
        self.downsample_factor = 0.25
        self.precomputed_hists = None
        self.current_pixel_size = 1
        self.dimensions = ("550px", "550px")
        self.current_plot_axis = 0
        self.rectangle_selector_on = False
        self.current_interval = 300
        self._init_fig()
        self._init_widgets()
        self._init_observes()
        self._init_links()
        self._init_app()

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

        self.rectangle_selector_button = Button(
            icon="far square",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Select a region of interest.",
        )
        # Rectangle selector
        self.rectangle_selector = bq.interacts.BrushSelector(
            x_scale=self.scale_x,
            y_scale=self.scale_y,
        )

        # Status bar updates
        self.status_bar_xrange = Label()
        self.status_bar_yrange = Label()
        self.status_bar_xrange.value = ""
        self.status_bar_yrange.value = ""
        self.status_bar_xdistance = Label()
        self.status_bar_ydistance = Label()
        self.status_bar_xdistance.value = ""
        self.status_bar_ydistance.value = ""
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

        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_pixel_range, "selected")

        # Rectangle selector button
        self.rectangle_selector_button.on_click(self.rectangle_select)

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
        self.image_index_slider.value = 0
        if self.downsample_factor == 1:
            self.plotted_image.image = self.original_imagestack[0]
            self.imagestack = self.original_imagestack
            self.change_aspect_ratio()
            return
        else:
            self.imagestack = rescale(
                self.imagestack,
                (1, self.downsample_factor, self.downsample_factor),
                anti_aliasing=False,
            )
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

    # Rectangle selector button
    def rectangle_select(self, change):
        if not self.rectangle_selector_on:
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
        if self.current_pixel_size is not None:
            self.nm_x = int(
                (self.pixel_range_x[1] - self.pixel_range_x[0])
                * self.current_pixel_size
            )
            self.nm_y = int(
                (self.pixel_range_y[1] - self.pixel_range_y[0])
                * self.current_pixel_size
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

    # Image plot

    # Intensity message
    def on_mouse_msg_intensity(self, interaction, data, buffers):
        if data["event"] == "mousemove":
            domain_x = data["domain"]["x"]
            domain_y = data["domain"]["y"]
            normalized_x = (domain_x - self.plotted_image.x[0]) / (
                self.plotted_image.x[1] - self.plotted_image.x[0]
            )
            normalized_y = (domain_y - self.plotted_image.y[0]) / (
                self.plotted_image.y[1] - self.plotted_image.y[0]
            )
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

    def _init_app(self):
        self.header_layout = Layout(justify_content="center", align_items="center")
        self.header = HBox(
            [
                self.downsample_viewer_button,
                self.downsample_viewer_textbox,
                self.scheme_dropdown,
            ],
            layout=self.header_layout,
        )
        self.center_layout = Layout(justify_content="center", align_content="center")
        self.center = HBox([self.fig, self.hist.fig], layout=self.center_layout)
        self.footer_layout = Layout(justify_content="center")
        self.footer1 = HBox(
            [self.play, self.image_index_slider], layout=self.footer_layout
        )
        self.init_buttons = [
            self.plus_button,
            self.minus_button,
            self.reset_button,
            self.rm_high_low_int_button,
            self.swap_axes_button,
            self.rectangle_selector_button,
            self.save_movie_button,
        ]
        self.all_buttons = self.init_buttons

    @abstractmethod
    def plot(self):
        ...

    @abstractmethod
    def create_app(self):
        ...


class BqImViewer_Import(BqImViewerBase):
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

    def plot(self, projections):
        self.projections = projections
        self.filedir = projections.filedir
        self.precomputed_hists = projections.hists
        self.current_pixel_size = projections.current_pixel_size
        self.pxX = projections.data.shape[2]
        self.pxY = projections.data.shape[1]
        self.original_imagestack = projections.data
        self.projections._check_downsampled_data()
        self.imagestack = np.array(self.projections.data_ds[0])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.change_downsample_button()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = np.min(self.imagestack)
        self.vmax = np.max(self.imagestack)
        self.image_index_slider.max = self.original_imagestack.shape[0] - 1
        self.image_index_slider.value = 0
        self.hist.preflatten_imagestack(self.imagestack)
        self.rm_high_low_int(None)


class BqImViewer_Prep(BqImViewer_Import):
    def __init__(self, Prep):
        self.Prep = Prep
        super().__init__()

    def create_app(self):
        # self.all_buttons.insert(-2, self.center_line_button)
        self.button_box = HBox(
            self.all_buttons,
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

    def plot(self):
        self.original_imagestack = self.Prep.Import.projections.data
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = np.array(self.Prep.Import.projections.data_ds[0])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.current_pixel_size = self.Prep.Import.projections.current_pixel_size
        self.precomputed_hists = self.Prep.Import.projections.hists
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


class BqImViewer_Import_Analysis(BqImViewer_Import):
    def __init__(self, Analysis):
        self.Analysis = Analysis
        super().__init__()
        # Rectangle selector button
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector. Select a region "
            "and copy it over to Altered Projections."
        )

    def plot(self):
        self.original_imagestack = self.Analysis.Import.projections.data
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = np.array(self.Analysis.Import.projections.data_ds[0])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.current_pixel_size = self.Analysis.Import.projections.current_pixel_size
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


class BqImViewer_Center(BqImViewer_Import_Analysis):
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
        self.all_buttons.insert(-2, self.center_line_button)
        self.button_box = HBox(
            self.all_buttons,
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
        self.imagestack = np.array(self.Center.Import.projections.data_ds[0])
        self.downsample_factor = 0.25
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.current_pixel_size = self.Center.Import.projections.current_pixel_size
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


class BqImViewer_Center_Recon(BqImViewer_Import):
    def __init__(self):
        super().__init__()

    def plot(self, rec):
        self.pxX = rec.shape[2]
        self.pxY = rec.shape[1]
        self.original_imagestack = rec
        self.imagestack = rec
        self.downsample_factor = 1
        self.downsample_viewer_textbox.value = self.downsample_factor
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


class BqImViewer_Altered_Analysis(BqImViewer_Import_Analysis):
    def __init__(self, viewer_parent, Analysis):
        self.Analysis = Analysis
        super().__init__(Analysis)
        self.viewer_parent = viewer_parent

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
        self.all_buttons.insert(-2, self.copy_button)
        self.all_buttons.insert(-2, self.link_plotted_projections_button)
        self.all_buttons.insert(-2, self.range_from_parent_button)
        self.button_box = HBox(
            self.all_buttons,
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
                    layout=Layout(justify_content="center"),
                ),
                HBox(
                    [self.remove_data_outside_button],
                    layout=Layout(justify_content="center"),
                ),
            ]
        )

        footer = VBox([self.footer1, footer2])

        self.app = VBox([self.header, self.center, footer])

    # Image plot
    def plot(self):
        self.original_imagestack = copy.copy(self.viewer_parent.original_imagestack)
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = copy.copy(self.viewer_parent.imagestack)
        self.pixel_range_x = [0, self.pxX - 1]
        self.pixel_range_y = [0, self.pxY - 1]
        self.pixel_range = [self.pixel_range_x, self.pixel_range_y]
        self.subset_pixel_range_x = self.pixel_range_x
        self.subset_pixel_range_y = self.pixel_range_y
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = self.viewer_parent.vmin
        self.vmax = self.viewer_parent.vmax
        self.image_scale["image"].min = self.vmin
        self.image_scale["image"].max = self.vmax
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0

    def copy_parent_projections(self, *args):
        self.copying = True
        self.plot()
        self.hist.hists = copy.copy(self.viewer_parent.hist.hists)
        self.hist.selector = bq.interacts.BrushIntervalSelector(
            orientation="vertical",
            scale=self.viewer_parent.hist.x_sc,
        )
        self.hist.selector.observe(self.hist.update_crange_selector, "selected")
        self.hist.fig.interaction = self.hist.selector
        self.hist.fig.marks = [self.hist.hists[0]]
        self.link_plotted_projections_button.button_style = "info"
        self.link_plotted_projections_button.disabled = False
        self.range_from_parent_button.disabled = False
        self.copying = False

    def link_plotted_projections(self, *args):
        if not self.plots_linked:
            self.plots_linked = True
            self.plot_link = jsdlink(
                (self.viewer_parent.image_index_slider, "value"),
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
            self.viewer_parent.rectangle_selector_button.button_style == "success"
            and self.viewer_parent.rectangle_selector.selected is not None
        ):
            imtemp = self.viewer_parent.imagestack
            lowerY = int(
                self.viewer_parent.pixel_range_y[0]
                * self.viewer_parent.downsample_factor
            )
            upperY = int(
                self.viewer_parent.pixel_range_y[1]
                * self.viewer_parent.downsample_factor
            )
            lowerX = int(
                self.viewer_parent.pixel_range_x[0]
                * self.viewer_parent.downsample_factor
            )
            upperX = int(
                self.viewer_parent.pixel_range_x[1]
                * self.viewer_parent.downsample_factor
            )
            self.imagestack = copy.deepcopy(imtemp[:, lowerY:upperY, lowerX:upperX])
            self.change_aspect_ratio()
            self.plotted_image.image = self.imagestack[
                self.viewer_parent.current_image_ind
            ]
            # This is confusing - decide on better names. The actual dimensions are
            # stored in self.projections.pixel_range_x, but this will eventually set the
            # Analysis attributes for pixel_range_x, pixel_range_y to input into
            # algorithms
            self.pixel_range_x = (
                self.viewer_parent.pixel_range_x[0],
                self.viewer_parent.pixel_range_x[1],
            )
            self.pixel_range_y = (
                self.viewer_parent.pixel_range_y[0],
                self.viewer_parent.pixel_range_y[1],
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
        self.original_imagestack[self.remove_high_indexes] = 1e-6
        self.remove_low_indexes = self.original_imagestack < self.vmin
        self.original_imagestack[self.remove_low_indexes] = 1e-6
        self.plotted_image.image = self.original_imagestack[0]
        self.remove_high_indexes = self.imagestack > self.vmax
        self.imagestack[self.remove_high_indexes] = 1e-6
        self.remove_low_indexes = self.imagestack < self.vmin
        self.imagestack[self.remove_low_indexes] = 1e-6
        self.plotted_image.image = self.imagestack[0]
        self.hist.preflatten_imagestack(self.imagestack)


class BqImViewer_TwoEnergy_High(BqImViewer_Import_Analysis):
    def __init__(self):
        super().__init__(None)
        # Rectangle selector button
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector. Select a region here "
            "to do phase correlation on."
        )
        self.rectangle_selector_on = False
        self.rectangle_select(None)
        self.viewing = False

    def plot(self, projections):
        self.projections = projections
        self.original_imagestack = projections.data
        self.filedir = projections.filedir
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = copy.deepcopy(self.original_imagestack)
        self.downsample_factor = 1
        self.downsample_viewer_textbox.value = self.downsample_factor
        self.current_pixel_size = projections.current_pixel_size
        self.precomputed_hists = projections.hists
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
        self.viewing = True
        self.change_buttons()

    def change_buttons(self):
        if self.viewer_child.viewing and self.viewing:
            self.viewer_child.scale_button.button_style = "info"
            self.viewer_child.scale_button.disabled = False
        else:
            self.viewer_child.scale_button.button_style = ""
            self.viewer_child.scale_button.disabled = True

    # Rectangle selector to update projection range
    def rectangle_to_pixel_range(self, *args):
        BqImViewer_Import.rectangle_to_pixel_range(self)
        self.viewer_child.match_rectangle_selector_range_parent()


class BqImViewer_TwoEnergy_Low(BqImViewer_TwoEnergy_High):
    def __init__(self, viewer_parent):
        super().__init__()
        self.viewer_parent = viewer_parent
        self.viewer_parent.viewer_child = self
        # Rectangle selector button
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector. Select a region here "
            "to do phase correlation on. This will be the moving image."
        )
        self.diff_button = Button(
            icon="minus",
            layout=self.button_layout,
            style=self.button_font,
            tooltip="Take the difference of the high and low energies.",
            disabled=True,
        )
        self.link_plotted_projections_button = Button(
            icon="unlink",
            tooltip="Link to the high energy slider.",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.link_plotted_projections_button.on_click(self.link_plotted_projections)
        self.plots_linked = False
        self.scale_button = Button(
            tooltip="Click this to scale the projections to the higher energy.",
            icon="compress",
            button_style="",
            disabled=True,
            layout=self.button_layout,
            style=self.button_font,
        )
        self.start_button = Button(
            disabled=True,
            button_style="",
            tooltip="Register low energy to high energy images",
            icon="fa-running",
            layout=self.button_layout,
            style=self.button_font,
        )

        self.diff_button.on_click(self.switch_to_diff)
        self._disable_diff_callback = True
        self.viewing = False
        self.diff_on = False

    # Rectangle selector to update projection range
    def rectangle_to_pixel_range(self, *args):
        BqImViewer_Import.rectangle_to_pixel_range(self)
        self.match_rectangle_selector_range_parent()

    def link_plotted_projections(self, *args):
        BqImViewer_DataExplorer_AfterAnalysis.link_plotted_projections(self)

    def match_rectangle_selector_range_parent(self):
        selected_x = self.rectangle_selector.selected_x
        selected_y = self.rectangle_selector.selected_y
        selected_x_par = self.viewer_parent.rectangle_selector.selected_x
        selected_y_par = self.viewer_parent.rectangle_selector.selected_y
        if selected_x is None:
            selected_x = selected_x_par
        if selected_y is None:
            selected_y = selected_y_par
        x_diff_par = selected_x_par[1] - selected_x_par[0]
        y_diff_par = selected_y_par[1] - selected_y_par[0]
        self.rectangle_selector.set_trait(
            "selected_x", [selected_x[0], selected_x[0] + x_diff_par]
        )
        self.rectangle_selector.set_trait(
            "selected_y", [selected_y[0], selected_y[0] + y_diff_par]
        )

    def change_buttons(self):
        if self.viewer_parent.viewing and self.viewing:
            self.scale_button.button_style = "info"
            self.scale_button.disabled = False
        else:
            self.scale_button.button_style = ""
            self.scale_button.disabled = True

    def create_app(self):
        self.all_buttons.insert(-2, self.diff_button)
        self.all_buttons.insert(-2, self.link_plotted_projections_button)
        self.all_buttons.insert(-2, self.scale_button)
        self.all_buttons.insert(-2, self.start_button)
        self.button_box = HBox(
            self.all_buttons,
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
                    layout=Layout(justify_content="center"),
                ),
            ]
        )

        footer = VBox([self.footer1, footer2])

        self.app = VBox([self.header, self.center, footer])

    def switch_to_diff(self, *args):
        if not self.diff_on and not self._disable_diff_callback:
            self.imagestack = self.diff_imagestack
            self.plotted_image.image = self.imagestack[self.image_index_slider.value]
            self.diff_on = True
            self._disable_diff_callback = True
            self.diff_button.button_style = "success"
            self._disable_diff_callback = False
        elif not self._disable_diff_callback:
            self.imagestack = self.original_imagestack
            self.plotted_image.image = self.imagestack[self.image_index_slider.value]
            self.diff_on = False
            self._disable_diff_callback = True
            self.diff_button.button_style = ""
            self._disable_diff_callback = False


class BqImViewer_DataExplorer_BeforeAnalysis(BqImViewer_Import):
    def __init__(self):
        super().__init__()

    def create_app(self):
        self.button_box = HBox(
            self.all_buttons,
            layout=self.footer_layout,
        )
        footer = VBox([self.footer1, self.button_box])
        self.app = VBox([self.header, self.center, footer])


class BqImViewer_DataExplorer_AfterAnalysis(BqImViewer_DataExplorer_BeforeAnalysis):
    def __init__(self, viewer_parent=None):
        super().__init__()
        self.viewer_parent = viewer_parent
        self.link_plotted_projections_button = Button(
            icon="unlink",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.link_plotted_projections_button.on_click(self.link_plotted_projections)
        self.plots_linked = False
        self.downsample_factor = 0.5

    def create_app(self):
        self.all_buttons = self.init_buttons
        self.all_buttons.insert(-2, self.link_plotted_projections_button)
        self.button_box = HBox(
            self.all_buttons,
            layout=self.footer_layout,
        )
        footer2 = VBox(
            [
                self.button_box,
            ]
        )

        footer = VBox([self.footer1, footer2])

        self.app = VBox([self.header, self.center, footer])

    def link_plotted_projections(self, *args):
        if not self.plots_linked:
            self.plot_link = jsdlink(
                (self.viewer_parent.image_index_slider, "value"),
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
        vmin_parent = self.viewer_parent.image_scale["image"].min
        vmax_parent = self.viewer_parent.image_scale["image"].max
        vmin = self.image_scale["image"].min
        vmax = self.image_scale["image"].max
        for i in range(len(self.original_imagestack)):
            im1 = ax1.imshow(
                self.viewer_parent.original_imagestack[i],
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


class BqImViewer_Altered_Prep(BqImViewer_Altered_Analysis):
    def __init__(self, viewer_parent, Prep):
        super().__init__(viewer_parent, Prep)
        self.Prep = Prep
        self.viewer_parent
        self.downsample_factor = 1

    def update_pixel_range_status_bar(self):
        self.status_bar_xrange.value = f"X Pixel Range: {self.printed_range_x} | "
        self.status_bar_yrange.value = f"Y Pixel Range: {self.printed_range_y}"

    # Rectangle selector button
    def rectangle_select(self, change):
        if self.rectangle_selector_on is False:
            self.fig.interaction = self.rectangle_selector
            self.fig.interaction.color = "magenta"
            self.rectangle_selector_on = True
            self.rectangle_selector_button.button_style = "success"
        else:
            self.rectangle_selector_button.button_style = ""
            self.rectangle_selector_on = False
            self.status_bar_xrange.value = ""
            self.status_bar_yrange.value = ""
            self.fig.interaction = self.msg_interaction

    def plot(self):
        if self.copying:
            self.Prep.prepped_data = copy.deepcopy(
                self.viewer_parent.original_imagestack
            )
        self.original_imagestack = self.Prep.prepped_data
        self.pxX = self.original_imagestack.shape[2]
        self.pxY = self.original_imagestack.shape[1]
        self.pxZ = self.original_imagestack.shape[0]
        self.imagestack = self.original_imagestack
        self.pixel_range_x = [0, self.pxX - 1]
        self.pixel_range_y = [0, self.pxY - 1]
        self.pixel_range = [self.pixel_range_x, self.pixel_range_y]
        self.subset_pixel_range_x = self.pixel_range_x
        self.subset_pixel_range_y = self.pixel_range_y
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.imagestack[0]
        self.vmin = self.viewer_parent.vmin
        self.vmax = self.viewer_parent.vmax
        self.image_scale["image"].min = self.vmin
        self.image_scale["image"].max = self.vmax
        self.image_index_slider.max = self.imagestack.shape[0] - 1
        self.image_index_slider.value = 0


class BqImHist:
    def __init__(self, implotter: BqImViewerBase):
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
            self.bin_centers = self.implotter.precomputed_hists[-1]["bin_centers"]
            self.frequency = self.implotter.precomputed_hists[-1]["frequency"]
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
        # current_pixel_size = 30
        # px_per_micron = 1000 / current_pixel_size
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
        #     marks=[altered_viewer.plotted_image, patch, test_label],
        #     animation_duration=1000,
        # )
        # fig.layout.height = dimensions[1]
        # fig.layout.width = dimensions[0]
        # fig
