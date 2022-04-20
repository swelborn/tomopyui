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

    def __init__(self, images=None):
        if images is None:
            self.original_images = np.random.rand(5, 50, 50)
            self.images = self.original_images
        else:
            self.original_images = images
            self.images = images
        self.current_image_ind = 0
        self.pxX = self.images.shape[2]
        self.pxY = self.images.shape[1]
        self.px_range_x = [0, self.pxX - 1]
        self.px_range_y = [0, self.pxY - 1]
        self.px_range = [self.px_range_x, self.px_range_y]
        self.px_size = 1
        self.aspect_ratio = self.pxX / self.pxY
        self.fig = None
        self.ds_factor = 0
        self.dimensions = ("550px", "550px")
        self.current_plot_axis = 0
        self.rectangle_selector_on = False
        self.current_interval = 300
        self.from_hdf = False
        self.from_npy = False
        self._init_fig()
        self._init_widgets()
        self._init_hist()
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
                min=np.min(self.images),
                max=np.max(self.images),
                scheme="viridis",
            ),
        }
        self.plotted_image = ImageGL(
            image=self.images[self.current_image_ind],
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
            max=self.images.shape[0] - 1,
            step=1,
        )

        # Image index play button
        self.play = Play(
            value=0,
            min=0,
            max=self.images.shape[0] - 1,
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

        # Downsample dropdown menu
        self.ds_viewer_dropdown = Dropdown(
            description="Viewer binning: ",
            options=[("Original", -1), ("2", 0), ("4", 1), ("8", 2)],
            value=0,
            style=extend_description_style,
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

    def _init_hist(self):
        # Histogram
        self.hist = BqImHist(self)

    def _init_observes(self):

        # Image index slider
        self.image_index_slider.observe(self.change_image, names="value")

        # Color scheme dropdown menu
        self.scheme_dropdown.observe(self.update_scheme, names="value")

        # Swap axes button
        self.swap_axes_button.on_click(self.swap_axes)

        # Faster play interval
        self.plus_button.on_click(self.speed_up)

        # Slower play interval
        self.minus_button.on_click(self.slow_down)

        # Save a movie at the current vmin/vmax
        self.save_movie_button.on_click(self.save_movie)

        # Downsample dropdown menu
        self.ds_viewer_dropdown.observe(self.downsample_viewer, "value")

        # Reset button
        self.reset_button.on_click(self.reset)

        # Zoom/intensity
        self.msg_interaction.on_msg(self.on_mouse_msg_intensity)

        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_px_range, "selected")

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
        self.plotted_image.image = self.images[change.new]
        self.current_image_ind = change.new

    # Scheme
    def update_scheme(self, *args):
        self.image_scale["image"].scheme = self.scheme_dropdown.value

    # Swap axes
    def swap_axes(self, *args):
        self.images = np.swapaxes(self.images, 0, 1)
        self.change_aspect_ratio()
        self.image_index_slider.max = self.images.shape[0] - 1
        self.image_index_slider.value = 0
        self.plotted_image.image = self.images[self.image_index_slider.value]
        if self.current_plot_axis == 0:
            self.current_plot_axis = 1
        else:
            self.current_plot_axis = 0

    # Downsample the plot view
    def downsample_viewer(self, *args):
        self.ds_factor = self.ds_viewer_dropdown.value
        if self.from_hdf and self.ds_factor != -1:
            self.projections._load_hdf_ds_data_into_memory(pyramid_level=self.ds_factor)
            self.original_images = self.projections.data
            self.images = self.projections.data_ds
            self.hist.precomputed_hist = self.projections.hist
        elif self.from_hdf and self.ds_factor == -1:
            self.projections._load_hdf_normalized_data_into_memory()
            self.original_images = self.projections.data
            self.images = self.projections.data
            self.hist.precomputed_hist = self.projections.hist
        else:
            if self.ds_factor == -1:
                self.plotted_image.image = self.original_images[0]
                self.images = self.original_images
                self.change_aspect_ratio()
            else:
                self.images = copy.deepcopy(self.original_images)
                ds_num = np.power(2, int(self.ds_factor) + 1)
                ds_num = float(1 / ds_num)
                self.images = rescale(
                    self.images,
                    (1, ds_num, ds_num),
                    anti_aliasing=False,
                )
        self.plotted_image.image = self.images[self.image_index_slider.value]
        self.change_aspect_ratio()

    # Reset
    def reset(self, *args):
        if self.current_plot_axis == 1:
            self.swap_axes()
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.plotted_image.image = self.images[0]
        self.hist.reset_state()
        self.image_scale["image"].min = self.hist.vmin
        self.image_scale["image"].max = self.hist.vmax
        self.image_index_slider.max = self.images.shape[0] - 1
        self.image_index_slider.value = 0

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
        if self.from_hdf:
            self.projections._load_hdf_ds_data_into_memory(
                self.ds_viewer_dropdown.value
            )
            self.images = self.projections.data_ds
        for image in self.images:
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
    def rectangle_to_px_range(self, *args):
        self.px_range = self.rectangle_selector.selected
        self.px_range = np.where(self.px_range < 0, 0, self.px_range)
        self.px_range = np.where(self.px_range > 1, 1, self.px_range)
        self.px_range_x = [
            int(x) for x in np.around(self.px_range[:, 0] * (self.pxX - 1))
        ]
        self.px_range_y = np.around(self.px_range[:, 1] * (self.pxY - 1))
        self.px_range_y = [int(x) for x in self.px_range_y]
        if self.px_size is not None:
            self.nm_x = int((self.px_range_x[1] - self.px_range_x[0]) * self.px_size)
            self.nm_y = int((self.px_range_y[1] - self.px_range_y[0]) * self.px_size)
            self.micron_x = round(self.nm_x / 1000, 2)
            self.micron_y = round(self.nm_y / 1000, 2)
        self.update_px_range_status_bar()

    def update_px_range_status_bar(self):
        self.status_bar_xrange.value = f"X Pixel Range: {self.px_range_x} | "
        self.status_bar_yrange.value = f"Y Pixel Range: {self.px_range_y}"
        if self.nm_x < 1000:
            self.status_bar_xdistance.value = f"X Distance (nm): {self.nm_x} | "
        else:
            self.status_bar_xdistance.value = f"X Distance (μm): {self.micron_x} | "
        if self.nm_y < 1000:
            self.status_bar_ydistance.value = f"Y Distance (nm): {self.nm_y}"
        else:
            self.status_bar_ydistance.value = f"Y Distance (μm): {self.micron_y}"

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
        pxX = self.images.shape[2]
        pxY = self.images.shape[1]
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

    def check_npy_or_hdf(self, projections):
        if projections.hdf_file is not None:
            self.from_hdf = True
            self.from_npy = False
        else:
            self.from_hdf = False
            self.from_npy = True

    def set_state_on_plot(self):
        self.pxX = self.original_images.shape[2]
        self.pxY = self.original_images.shape[1]
        self.pxZ = self.original_images.shape[0]
        self.px_range_x = [0, self.pxX - 1]
        self.px_range_y = [0, self.pxY - 1]
        self.px_range = [self.px_range_x, self.px_range_y]
        self.plotted_image.image = self.images[0]
        self.image_index_slider.max = self.pxZ - 1
        self.image_index_slider.value = 0
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.hist.refresh_histogram()
        self.projections._close_hdf_file()

    def get_ds_factor_from_dropdown(self):
        ds_factor = self.ds_viewer_dropdown.value
        if ds_factor == -1:
            ds_factor = 1
        else:
            ds_factor = np.power(2, int(ds_factor) + 1)
        ds_factor = 1 / ds_factor
        return ds_factor

    def _init_app(self):
        self.header_layout = Layout(justify_content="center", align_items="center")
        self.header = HBox(
            [
                self.ds_viewer_dropdown,
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
    def plot(self, io_obj):
        ...

    @abstractmethod
    def create_app(self):
        ...


class BqImViewer_Projections_Parent(BqImViewerBase):
    def __init__(self):
        super().__init__()
        self.rectangle_selector_button.tooltip = (
            "Turn on the rectangular region selector. Select a region "
            "and copy it over to Altered Projections."
        )

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

    def plot(self, projections, ds=True):
        self.projections = projections
        self.filedir = self.projections.filedir
        self.px_size = self.projections.px_size
        self.check_npy_or_hdf(projections)
        if ds is True:
            self.projections._check_downsampled_data()
            self.ds_viewer_dropdown.value = (
                0 if any([0 == x[1] for x in self.ds_viewer_dropdown.options]) else -1
            )
            self.hist.precomputed_hist = self.projections.hist
            self.original_images = self.projections.data
            self.images = self.projections.data_ds
        else:
            self.ds_viewer_dropdown.value = -1
            self.ds_viewer_dropdown.options = [("Original", -1)]
            self.original_images = self.projections.data
            self.images = self.original_images
        self.set_state_on_plot()


class BqImViewer_Projections_Child(BqImViewer_Projections_Parent):
    def __init__(self, viewer_parent):
        self.viewer_parent = viewer_parent
        super().__init__()
        self.subset_x = None
        self.subset_y = None
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

        # Rectangle selector
        self.rectangle_selector.observe(self.rectangle_to_px_range, "selected")
        self.all_buttons.insert(-2, self.copy_button)
        self.all_buttons.insert(-2, self.link_plotted_projections_button)
        self.all_buttons.insert(-2, self.range_from_parent_button)

    def _init_hist(self):
        self.hist = BqImHist_Child(self, self.viewer_parent)

    def create_app(self):
        self.center = HBox([self.fig, self.hist.fig], layout=self.center_layout)
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

    def plot(self, projections, ds=True):
        super().plot(projections, ds=ds)
        self.rm_high_low_int_button.disabled = False
        self.subset_px_range_x = self.px_range_x
        self.subset_px_range_y = self.px_range_y

    def copy_parent_projections(self, *args):
        self.copying = True
        self.projections.parent_projections = self.viewer_parent.projections
        self.projections.copy_from_parent()
        self.hist.copy_parent_hist()
        self.plot(self.projections)
        self.hist.refresh_histogram()
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
            ds_factor = self.viewer_parent.get_ds_factor_from_dropdown()
            imtemp = self.viewer_parent.images
            lowerY = int(self.viewer_parent.px_range_y[0] * ds_factor)
            upperY = int(self.viewer_parent.px_range_y[1] * ds_factor)
            lowerX = int(self.viewer_parent.px_range_x[0] * ds_factor)
            upperX = int(self.viewer_parent.px_range_x[1] * ds_factor)
            self.images = copy.deepcopy(imtemp[:, lowerY:upperY, lowerX:upperX])
            self.change_aspect_ratio()
            self.plotted_image.image = self.images[self.viewer_parent.current_image_ind]
            # This is confusing - decide on better names. The actual dimensions are
            # stored in self.projections.px_range_x, but this will eventually set the
            # Analysis attributes for px_range_x, px_range_y to input into
            # algorithms
            self.px_range_x = (
                self.viewer_parent.px_range_x[0],
                self.viewer_parent.px_range_x[1],
            )
            self.px_range_y = (
                self.viewer_parent.px_range_y[0],
                self.viewer_parent.px_range_y[1],
            )

    # Rectangle selector to update projection range
    def rectangle_to_px_range(self, *args):
        self.px_range = self.rectangle_selector.selected
        x_len = self.px_range_x[1] - self.px_range_x[0]
        y_len = self.px_range_y[1] - self.px_range_y[0]
        lowerX = int(self.px_range[0, 0] * x_len + self.px_range_x[0])
        upperX = int(self.px_range[1, 0] * x_len + self.px_range_x[0])
        lowerY = int(self.px_range[0, 1] * y_len + self.px_range_y[0])
        upperY = int(self.px_range[1, 1] * y_len + self.px_range_y[0])
        self.printed_range_x = [lowerX, upperX]
        self.printed_range_y = [lowerY, upperY]
        self.subset_x = [x - self.px_range_x[0] for x in self.printed_range_x]
        self.subset_y = [y - self.px_range_y[0] for y in self.printed_range_y]
        self.update_px_range_status_bar()

    # Rectangle selector button
    def rectangle_select(self, change):
        if self.rectangle_selector_on is False:
            self.fig.interaction = self.rectangle_selector
            self.fig.interaction.color = "magenta"
            self.rectangle_selector_on = True
            self.rectangle_selector_button.button_style = "success"
            # self.Analysis.use_subset_correlation_checkbox.value = True
        else:
            self.rectangle_selector_button.button_style = ""
            self.rectangle_selector_on = False
            self.status_bar_xrange.value = ""
            self.status_bar_yrange.value = ""
            self.fig.interaction = self.msg_interaction
            # self.Analysis.use_subset_correlation_checkbox.value = False

    def update_px_range_status_bar(self):
        self.status_bar_xrange.value = (
            f"Phase Correlation X Pixel Range: {self.printed_range_x} | "
        )
        self.status_bar_yrange.value = (
            f"Phase Correlation Y Pixel Range: {self.printed_range_y}"
        )


class BqImViewer_Center(BqImViewer_Projections_Parent):
    def __init__(self):
        super().__init__()
        self.center_line_on = True
        self.center_line = bq.Lines(
            x=[0.5, 0.5],
            y=[0, 1],
            colors="red",
            stroke_width=3,
            scales={"x": self.scale_x, "y": self.scale_y},
        )
        self.slice_line = bq.Lines(
            x=[0, 1],
            y=[0.5, 0.5],
            colors="green",
            stroke_width=3,
            scales={"x": self.scale_x, "y": self.scale_y},
        )
        self.slice_line_slider = IntSlider(
            value=0,
            min=0,
            max=self.images.shape[0] - 1,
            step=1,
        )
        self.fig.marks = (
            self.plotted_image,
            self.center_line,
            self.slice_line,
        )

    def create_app(self):
        self.button_box = HBox(
            self.all_buttons,
            layout=self.footer_layout,
        )
        self.footer2 = VBox(
            [
                HBox([self.slice_line_slider], layout=Layout(justify_content="center")),
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
        self.footer = VBox([self.footer1, self.footer2])
        self.app = VBox([self.header, self.center, self.footer])

    def plot(self, projections):
        super().plot(projections)
        self.slice_line_slider.max = self.pxY - 1
        self.slice_line_slider.value = int((self.pxY - 1) / 2)


class BqImViewer_Center_Recon(BqImViewer_Projections_Parent):
    # TODO: make reconstruction io_object
    def plot(self, rec):
        self.pxX = rec.shape[2]
        self.pxY = rec.shape[1]
        self.original_images = rec
        self.images = rec
        self.ds_viewer_dropdown.value = 0
        self.ds_factor = self.ds_viewer_dropdown.value
        self.current_image_ind = 0
        self.change_aspect_ratio()
        self.image_index_slider.max = self.images.shape[0] - 1
        self.image_index_slider.value = int(self.images.shape[0] / 2)
        self.plotted_image.image = self.images[self.image_index_slider.value]
        self.hist.refresh_histogram(self.images)
        self.hist.rm_high_low_int(None)


class BqImViewer_TwoEnergy_High(BqImViewer_Projections_Parent):
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

    def plot(self, projections):
        super().plot(projections)
        self.filedir = projections.filedir
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
    def rectangle_to_px_range(self, *args):
        BqImViewer_Projections_Parent.rectangle_to_px_range(self)
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

        self.all_buttons.insert(-2, self.diff_button)
        self.all_buttons.insert(-2, self.link_plotted_projections_button)
        self.all_buttons.insert(-2, self.scale_button)
        self.all_buttons.insert(-2, self.start_button)
        self.diff_button.on_click(self.switch_to_diff)
        self._disable_diff_callback = True
        self.viewing = False
        self.diff_on = False

    # Rectangle selector to update projection range
    def rectangle_to_px_range(self, *args):
        BqImViewer_Projections_Parent.rectangle_to_px_range(self)
        self.match_rectangle_selector_range_parent()

    def link_plotted_projections(self, *args):
        BqImViewer_Projections_Child.link_plotted_projections(self)

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
            self.images = self.diff_images
            self.plotted_image.image = self.images[self.image_index_slider.value]
            self.diff_on = True
            self._disable_diff_callback = True
            self.diff_button.button_style = "success"
            self._disable_diff_callback = False
        elif not self._disable_diff_callback:
            self.images = self.original_images
            self.plotted_image.image = self.images[self.image_index_slider.value]
            self.diff_on = False
            self._disable_diff_callback = True
            self.diff_button.button_style = ""
            self._disable_diff_callback = False


class BqImHist:
    def __init__(self, viewer: BqImViewerBase):
        self.vmin = np.min(viewer.images)
        self.vmax = np.max(viewer.images)
        self.init_vmin = None
        self.init_vmax = None
        self.viewer = viewer
        self.viewer.rm_high_low_int_button.on_click(self.rm_high_low_int)
        self.fig = bq.Figure(
            padding=0,
            fig_margin=dict(top=0, bottom=0, left=0, right=0),
        )
        self.precomputed_hist = None
        self.refresh_histogram(self.viewer.images)
        self.fig.layout.width = "100px"
        self.fig.layout.height = viewer.fig.layout.height
        self.copied_hist = False

    def reset_state(self):
        self.vmin = self.init_vmin
        self.vmax = self.init_vmax
        self.selector.selected = None
        self.vmin = np.min(self.images)
        self.vmax = np.max(self.images)
        self.rm_high_low_int(None)

    def refresh_histogram(self, images=None):
        if self.precomputed_hist is not None:
            if self.viewer.from_hdf or self.copied_hist:
                self.refresh_histogram_from_hdf()
            else:
                self.refresh_histogram_from_downsampled_folder()
            self.x_sc = bq.LinearScale(
                min=float(self.images_min), max=float(self.images_max)
            )
            self.y_sc = bq.LinearScale()
            self.fig.scale_x = self.x_sc
            self.fig.scale_y = bq.LinearScale()
            self.hist = bq.Bars(
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
            ind = self.bin_centers < self.vmin
            self.frequency[ind] = 0
            self.hist.scales["y"].max = float(np.max(self.frequency))

        else:
            self.refresh_histogram_without_precompute()

        self.selector = bq.interacts.BrushIntervalSelector(
            orientation="vertical", scale=self.x_sc
        )
        self.selector.observe(self.update_crange_selector, "selected")
        self.fig.marks = [self.hist]
        self.fig.interaction = self.selector
        self.rm_high_low_int(None)

    def refresh_histogram_from_hdf(self):
        self.ds_factor = self.viewer.ds_factor
        self.ds_dict = self.precomputed_hist
        self.bin_centers = self.precomputed_hist[
            self.viewer.projections.hdf_key_bin_centers
        ]
        self.frequency = self.ds_dict[self.viewer.projections.hdf_key_bin_frequency]
        self.images_min = float(
            self.ds_dict[self.viewer.projections.hdf_key_image_range][0]
        )
        self.images_max = float(
            self.ds_dict[self.viewer.projections.hdf_key_image_range][1]
        )
        self.vmin = float(self.ds_dict[self.viewer.projections.hdf_key_percentile][0])
        self.init_vmin = float(self.vmin)
        self.vmax = float(self.ds_dict[self.viewer.projections.hdf_key_percentile][1])
        self.init_vmax = float(self.vmax)
        self.ds_factor_num = self.ds_dict[self.viewer.projections.hdf_key_ds_factor]

    def refresh_histogram_from_downsampled_folder(self):
        self.bin_centers = self.precomputed_hist[-1]["bin_centers"]
        self.frequency = self.precomputed_hist[-1]["frequency"]
        self.images_min = float(np.min(self.viewer.images))
        self.images_max = float(np.max(self.viewer.images))
        self.vmin = self.images_min
        self.vmax = self.images_max

    def refresh_histogram_without_precompute(self):
        self.images_min = float(np.min(self.viewer.images))
        self.images_max = float(np.max(self.viewer.images))
        self.vmin = self.images_min
        self.vmax = self.images_max
        self.x_sc = bq.LinearScale(min=float(self.vmin), max=float(self.vmax))
        self.y_sc = bq.LinearScale()
        self.fig.scale_x = self.x_sc
        self.fig.scale_y = bq.LinearScale()
        self.hist = bq.Bins(
            sample=self.viewer.images.ravel(),
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
        self.bin_centers = self.hist.x
        self.frequency = self.hist.y
        ind = self.bin_centers < self.vmin
        self.frequency[ind] = 0
        self.hist.scales["y"].max = np.max(self.frequency)

    def update_crange_selector(self, *args):
        if self.selector.selected is not None:
            self.viewer.image_scale["image"].min = self.selector.selected[0]
            self.viewer.image_scale["image"].max = self.selector.selected[1]
            self.vmin = self.selector.selected[0]
            self.vmax = self.selector.selected[1]

    def rm_high_low_int(self, change):
        if self.viewer.from_hdf or self.viewer.from_npy:
            self.vmin, self.vmax = self.init_vmin, self.init_vmax
            self.selector.selected = [float(self.vmin), float(self.vmax)]
        else:
            self.vmin, self.vmax = np.percentile(self.viewer.images, q=(0.5, 99.5))
            self.selector.selected = [self.vmin, self.vmax]


class BqImHist_Child(BqImHist):
    def __init__(self, viewer, viewer_parent):
        super().__init__(viewer)
        self.viewer_parent = viewer_parent

    def copy_parent_hist(self):
        self.precomputed_hist = copy.copy(self.viewer_parent.hist.precomputed_hist)
        self.copied_hist = True
        # self.selector = bq.interacts.BrushIntervalSelector(
        #     orientation="vertical",
        #     scale=self.viewer_parent.hist.x_sc,
        # )
        # self.selector.observe(self.update_crange_selector, "selected")
        # self.fig.interaction = self.selector
        # self.fig.marks = [self.hist]


class ScaleBar:
    def __init__(self):
        pass

        # attempt to make scale bar. try again later 02/05/2022
        # import bqplot as bq

        # sc_x = LinearScale(min=0, max=1)
        # sc_y = LinearScale(min=1, max=0)

        # pxX = 1024
        # px_size = 30
        # px_per_micron = 1000 / px_size
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
