import numpy as np
import copy

# astra_cuda_recon_algorithm_kwargs, tomopy_recon_algorithm_kwargs,
# and tomopy_filter_names, extend_description_style
from tomopyui._sharedvars import *
from ipywidgets import *
from tomopy.recon.rotation import find_center_vo, find_center, find_center_pc
from tomopyui.widgets.view import BqImViewer_Center, BqImViewer_Center_Recon
from tomopyui.backend.util.center import write_center
from tomopyui.widgets.helpers import ReactiveTextButton, ReactiveIconButton
from scipy.stats import linregress


class Center:
    """
    Class for creating a tab to help find the center of rotation. See examples
    for more information on center finding.

    Attributes
    ----------
    Import : `Import`
        Needs an import object to be constructed.
    current_center : double
        Current center of rotation. Updated when center_textbox is updated.
    center_guess : double
        Guess value for center of rotation for automatic alignment (`~tomopy.recon.rotation.find_center`).
    index_to_try : int
        Index to try out when automatically (entropy) or manually trying to
        find the center of rotation.
    search_step : double
        Step size between centers (see `tomopy.recon.rotation.write_center` or
        `tomopyui.backend.util.center`).
    search_range : double
        Will search from [center_guess - search_range] to [center_guess + search range]
        in steps of search_step.
    num_iter : int
        Number of iterations to use in center reconstruction.
    algorithm : str
        Algorithm to use in the reconstruction. Chosen from dropdown list.
    filter : str
        Filter to be used. Only works with fbp and gridrec. If you choose
        another algorith, this will be ignored.

    """

    def __init__(self, Import):
        self.Import = Import
        self.projections = Import.projections
        self.Import.Center = self
        self.current_center = self.Import.prenorm_uploader.projections.pxX / 2
        self.center_slice_list = []
        self.use_multiple_centers = False
        self.center_guess = None
        self.index_to_try = None
        self.search_step = 10
        self.search_range = 100
        self.recon_slice = None
        self.cen_range = None
        self.use_ds = True
        self.num_iter = int(1)
        self.algorithm = "gridrec"
        self.filter = "parzen"
        self.metadata = {}
        self.viewer = BqImViewer_Center()
        self.viewer.create_app()
        self.rec_viewer = BqImViewer_Center_Recon()
        self.rec_viewer.create_app()
        self.reg = None
        self.header_font_style = {
            "font_size": "22px",
            "font_weight": "bold",
            "font_variant": "small-caps",
        }
        self.button_font = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")
        self._init_widgets()
        self._set_observes()
        self.make_tab()

    def set_metadata(self):
        """
        Sets `Center` metadata.
        """
        self.metadata["center"] = self.current_center
        self.metadata["center_guess"] = self.center_guess
        self.metadata["index_to_try"] = self.index_to_try
        self.metadata["search_step"] = self.search_step
        self.metadata["search_range"] = self.search_range
        self.metadata["cen_range"] = self.cen_range
        self.metadata["num_iter"] = self.num_iter
        self.metadata["algorithm"] = self.algorithm
        self.metadata["filter"] = self.filter

    def _init_widgets(self):

        self.center_textbox = FloatText(
            description="Center: ",
            disabled=False,
            style=extend_description_style,
            layout=Layout(justify_content="center")
        )
        self.load_rough_center = Button(
            description="Click to load rough center from imported data.",
            disabled=False,
            button_style="info",
            tooltip="Loads the half-way pixel point for the center.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.center_guess_textbox = FloatText(
            description="Guess for center: ",
            disabled=False,
            style=extend_description_style,
        )
        self.find_center_button = ReactiveTextButton(
            self.find_center_on_click,
            "Click to automatically find center (image entropy).",
            "Automatically finding center (image entropy).",
            "Found center.",
            warning="Please import some data first.",
        )
        self.find_center_vo_button = ReactiveTextButton(
            self.find_center_vo_on_click,
            "Click to automatically find center (Vo).",
            "Automatically finding center.",
            "Found center.",
            warning="Please import some data first.",
        )
        self.find_center_manual_button = ReactiveTextButton(
            self.find_center_manual_on_click,
            "Click to find center by plotting.",
            "Reconstructing.",
            "Now search for the center using the reconstruction slider. Add more values if your sample has multiple centers of rotation.",
            warning="Your projections do not have associated theta values.",
        )
        self.index_to_try_textbox = IntText(
            description="Slice to use: ",
            disabled=False,
            style=extend_description_style,
            placeholder="Default is 1/2*y pixels",
        )
        self.num_iter_textbox = IntText(
            description="Number of iterations: ",
            disabled=False,
            style=extend_description_style,
            value=self.num_iter,
        )
        self.search_range_textbox = IntText(
            description="Search range around center:",
            disabled=False,
            style=extend_description_style,
            value=self.search_range,
        )
        self.search_step_textbox = FloatText(
            description="Step size in search range: ",
            disabled=False,
            style=extend_description_style,
            value=self.search_step,
        )
        self.algorithms_dropdown = Dropdown(
            options=[key for key in tomopy_recon_algorithm_kwargs],
            value=self.algorithm,
            description="Algorithm:",
        )
        self.filters_dropdown = Dropdown(
            options=[key for key in tomopy_filter_names],
            value=self.filter,
            description="Filter:",
        )
        self.use_ds_checkbox = Checkbox(
            description="Use viewer downsampling: ",
            value=True,
            disabled=True,
            style=extend_description_style,
        )
        self.projections_plot_header = "Projections"
        self.projections_plot_header = Label(
            self.projections_plot_header, style=self.header_font_style
        )
        self.reconstructions_plot_header = "Reconstruction"
        self.reconstructions_plot_header = Label(
            self.reconstructions_plot_header, style=self.header_font_style
        )

        # -- Centers display -----------------------------------------------------------
        self.center_select_label = Label("Slice : Center", style=self.header_font_style)
        self.center_select = Select(
            disabled=True,
            rows=10,
        )
        self.all_centers_select_label = Label(
            "Centers for reconstruction", style=self.header_font_style
        )
        self.all_centers_select = Select(
            disabled=True,
            rows=10,
        )
        self.remove_center_button = Button(
            disabled=True,
            icon="fa-minus-square",
            tooltip="Remove selected center.",
            layout=self.button_layout,
            style=self.button_font,
        )
        self.buttons_to_disable = [
            self.center_select,
            self.remove_center_button,
        ]
        self.add_center_button = ReactiveIconButton(
            callback=self.set_center_to_slice,
            icon="fa-plus-square",
            tooltip="Add center from reconstruction of this slice.",
            skip_during=True,
            layout=self.button_layout,
            style=self.button_font,
            disabled=False,
        )
        self.add_center_button.button.disabled = True
        self.rec_viewer.image_index_slider.disabled = True

    def _use_ds_data(self, change):
        self.use_ds = self.use_ds_checkbox.value

    def _center_update(self, change):
        self.current_center = change.new
        self.center_guess = change.new
        self.viewer.center_line.x = [
            change.new / self.viewer.pxX,
            change.new / self.viewer.pxX,
        ]
        self.set_metadata()

    def _center_guess_update(self, change):
        self.center_guess = change.new
        self.viewer.center_line.x = [
            change.new / self.viewer.pxX,
            change.new / self.viewer.pxX,
        ]
        self.set_metadata()

    def _load_rough_center_onclick(self, change):
        self.center_guess = self.projections.pxX / 2
        self.current_center = self.center_guess
        self.center_textbox.value = self.center_guess
        self.center_guess_textbox.value = self.center_guess
        self.index_to_try_textbox.value = int(
            np.around(self.Import.projections.pxY / 2)
        )
        self.index_to_try = self.index_to_try_textbox.value
        self.set_metadata()

    def _index_to_try_update(self, change):
        self.index_to_try = change.new
        self.set_metadata()

    def _num_iter_update(self, change):
        self.num_iter = change.new
        self.set_metadata()

    def _search_range_update(self, change):
        self.search_range = change.new
        self.set_metadata()

    def _search_step_update(self, change):
        self.search_step = change.new
        self.set_metadata()

    def _update_algorithm(self, change):
        self.algorithm = change.new
        self.set_metadata()

    def _update_filters(self, change):
        self.filter = change.new
        self.set_metadata()

    def _slice_slider_update(self, change):
        slider_ind = change.new
        line_display = self.viewer.pxY - slider_ind
        self.viewer.slice_line.y = [
            line_display / self.viewer.pxY,
            line_display / self.viewer.pxY,
        ]
        self.index_to_try_textbox.value = int(line_display)

    def _center_textbox_slider_update(self, change):
        if self.add_center_button.button.button_style == "success":
            self.add_center_button.reset_state()
        self.center_textbox.value = self.cen_range[change.new]
        self.center_guess_textbox.value = self.cen_range[change.new]
        self.current_center = self.center_textbox.value
        self.viewer.update_center_line(self, change.new)
        self.set_metadata()

    def set_center_to_slice(self, *args):
        add_cond = [self.recon_slice == x[1] for x in self.center_slice_list]
        if any(add_cond):
            ind = [i for i, j in enumerate(add_cond) if j]
            [self.center_slice_list.pop(i) for i in ind]
        center_slice = (self.current_center, self.recon_slice)
        self.center_slice_list.append(center_slice)
        self.update_center_select()

    def remove_center(self, *args):
        ind = self.center_select.index
        if ind is None:
            self.center_slice_list.pop(-1)
        else:
            self.center_slice_list.pop(ind)
        self.update_center_select()

    def update_center_select(self):
        slice_column = range(self.projections.pxY)
        if len(self.center_slice_list) > 1:
            centers = [cen[0] for cen in self.center_slice_list]
            slices = [cen[1] for cen in self.center_slice_list]
            self.reg = linregress(slices, centers)
            self.reg_centers = [
                self.reg.slope * x + self.reg.intercept for x in slice_column
            ]

        elif len(self.center_slice_list) == 1:
            self.reg = None
            self.reg_centers = [self.current_center for i in slice_column]
        if self.center_slice_list == []:
            self.center_select.options = ["No centers set"]
            self.reg_centers = ["None" for i in slice_column]
            for x in self.buttons_to_disable:
                x.disabled = True
            self.all_centers_select.options = [
                f"{x}" + " : " + y for x, y in zip(slice_column, self.reg_centers)
            ]
        else:
            self.center_select.options = [
                f"{x[1]} : {x[0]:0.1f}" for x in self.center_slice_list
            ]
            for x in self.buttons_to_disable:
                x.disabled = False
            self.all_centers_select.options = [
                f"{x} : {y:0.1f}" for x, y in zip(slice_column, self.reg_centers)
            ]
        self.viewer.update_tilted_center_line(self)

    def _set_observes(self):
        self.center_textbox.observe(self._center_update, names="value")
        self.center_guess_textbox.observe(self._center_guess_update, names="value")
        self.load_rough_center.on_click(self._load_rough_center_onclick)
        self.index_to_try_textbox.observe(self._index_to_try_update, names="value")
        self.num_iter_textbox.observe(self._num_iter_update, names="value")
        self.search_range_textbox.observe(self._search_range_update, names="value")
        self.search_step_textbox.observe(self._search_step_update, names="value")
        self.algorithms_dropdown.observe(self._update_algorithm, names="value")
        self.filters_dropdown.observe(self._update_filters, names="value")
        # Callback for index going to center
        self.rec_viewer.image_index_slider.observe(
            self._center_textbox_slider_update, names="value"
        )
        self.viewer.slice_line_slider.observe(self._slice_slider_update, names="value")
        self.use_ds_checkbox.observe(self._use_ds_data, names="value")
        self.remove_center_button.on_click(self.remove_center)

    def find_center_on_click(self, *args):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center`. Takes index_to_try and center_guess.
        This method has worked better for me, if I use a good index_to_try
        and center_guess.
        """
        prj_imgs, ds_value = self.get_ds_projections()
        angles_rad = self.projections.angles_rad
        self.current_center = find_center(
            prj_imgs,
            angles_rad,
            ratio=0.9,
            ind=self.index_to_try,
            init=self.center_guess,
        )
        self.center_textbox.value = self.current_center

    def find_center_vo_on_click(self, *args):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center_vo`. Note: this method has not worked
        well for me.
        """
        prj_imgs, ds_value = self.get_ds_projections()
        angles_rad = self.projections.angles_rad
        try:
            self.current_center = find_center_vo(prj_imgs, ncore=1)
            self.center_textbox.value = self.current_center
        except Excepton:
            self.find_center_vo_button.warning()

    def find_center_manual_on_click(self, *args):
        """
        Reconstructs at various centers when you click the button, and plots
        the results with a slider so one can view. TODO: see X example.
        Uses search_range, search_step, center_guess.
        Creates a :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` +
        :doc:`histogram <mpl-interactions:examples/hist>` plot
        """
        self.add_center_button.reset_state()
        if self.rec_viewer.image_index_slider.disabled:
            self.rec_viewer.image_index_slider.disabled = False
        self.recon_slice = copy.deepcopy(self.index_to_try)
        prj_imgs, ds_value = self.get_ds_projections()
        angles_rad = self.projections.angles_rad
        ds_factor = np.power(2, int(ds_value + 1))
        _center_guess = copy.deepcopy(self.center_guess) / ds_factor
        _search_range = copy.deepcopy(self.search_range) / ds_factor
        _search_step = copy.deepcopy(self.search_step) / ds_factor
        _index_to_try = int(copy.deepcopy(self.index_to_try) / ds_factor)

        cen_range = [
            _center_guess - _search_range,
            _center_guess + _search_range,
            _search_step,
        ]
        # reconstruct, but also pull the centers used out to map to center
        # textbox
        self.rec, cen_range = write_center(
            prj_imgs,
            angles_rad,
            cen_range=cen_range,
            ind=_index_to_try,
            mask=True,
            algorithm=self.algorithm,
            filter_name=self.filter,
            num_iter=self.num_iter,
        )
        self.cen_range = [ds_factor * cen for cen in cen_range]
        if self.rec is None:
            self.find_center_manual_button.warning()
            return
        self.rec_viewer.plot(self.rec)
        self.add_center_button.button.disabled = False

    def get_ds_projections(self):
        ds_value = self.viewer.ds_viewer_dropdown.value
        if self.use_ds:
            if ds_value == -1:
                prj_imgs = self.projections.data
            else:
                self.projections._load_hdf_ds_data_into_memory(pyramid_level=ds_value)
                prj_imgs = self.projections.data_ds
        elif ds_value == -1:
            prj_imgs = self.projections.data
        else:
            self.projections._load_hdf_normalized_data_into_memory()
            prj_imgs = self.projections.data
        return prj_imgs, ds_value

    def refresh_plots(self):
        self.viewer.plot(self.projections, no_check=True)
        self.reg = None
        self._load_rough_center_onclick(None)
        self.find_center_button.enable()
        self.find_center_manual_button.enable()
        self.find_center_vo_button.enable()

    def make_tab(self):
        """
        Function to create a Center object's :doc:`Tab <ipywidgets:index>`.
        """

        # Accordion to find center automatically
        self.automatic_center_vbox = VBox(
            [
                HBox(
                    [self.find_center_button.button, self.find_center_vo_button.button],
                    layout=Layout(justify_content="center"),
                ),
                HBox(
                    [
                        self.center_guess_textbox,
                        self.index_to_try_textbox,
                    ],
                    layout=Layout(justify_content="center"),
                ),
            ]
        )
        self.automatic_center_accordion = Accordion(
            children=[self.automatic_center_vbox],
            selected_index=None,
            titles=("Find center automatically",),
        )

        self.viewer_hbox = HBox(
            [
                VBox(
                    [
                        self.projections_plot_header,
                        self.viewer.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
                VBox(
                    [
                        self.reconstructions_plot_header,
                        self.rec_viewer.app,
                    ],
                    layout=Layout(align_items="center"),
                ),
            ],
            layout=Layout(justify_content="center", align_items="center"),
        )
        # Accordion to find center manually
        self.manual_center_vbox = VBox(
            [
                self.viewer_hbox,
                HBox([self.center_textbox],layout=Layout(justify_content="center")),
                HBox(
                    [
                        self.find_center_manual_button.button,
                        self.add_center_button.button,
                        self.remove_center_button,
                    ],
                    layout=Layout(justify_content="center"),
                ),
                HBox(
                    [
                        HBox(
                            [
                                self.center_guess_textbox,
                                self.index_to_try_textbox,
                                self.num_iter_textbox,
                                self.search_range_textbox,
                                self.search_step_textbox,
                                self.algorithms_dropdown,
                                self.filters_dropdown,
                                self.use_ds_checkbox,
                            ],
                            layout=Layout(
                                width="50%",
                                flex_flow="row wrap",
                                justify_content="center",
                                align_items="center",
                            ),
                        ),
                        HBox(
                            [
                                VBox(
                                    [
                                        self.center_select_label,
                                        self.center_select,
                                    ],
                                    layout=Layout(align_items="center"),
                                ),
                                VBox(
                                    [
                                        self.all_centers_select_label,
                                        self.all_centers_select,
                                    ],
                                    layout=Layout(align_items="center"),
                                ),
                            ],
                        ),
                    ]
                ),
            ],
            layout=Layout(justify_content="center"),
        )

        self.manual_center_accordion = Accordion(
            children=[self.manual_center_vbox],
            selected_index=0,
            titles=("Find center manually",),
        )

        self.tab = VBox(
            [
                VBox(
                    [
                        HBox(
                            [
                                self.Import.switch_data_buttons,
                                self.load_rough_center,
                            ]
                        ),
                    ],
                ),
                self.manual_center_accordion,
                self.automatic_center_accordion,
            ]
        )
