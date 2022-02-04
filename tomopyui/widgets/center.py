from ipywidgets import *

# includes astra_cuda_recon_algorithm_kwargs, tomopy_recon_algorithm_kwargs,
# and tomopy_filter_names, extend_description_style
from tomopyui._sharedvars import *
import numpy as np
from tomopy.recon.rotation import find_center_vo, find_center, find_center_pc
from tomopyui.widgets.plot import BqImPlotter_Center
from tomopyui.backend.util.center import write_center


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

        TODO: this should be linked to both `Align` and `Recon`.
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
        self.current_center = self.Import.prenorm_projections.pxX / 2
        self.center_guess = None
        self.index_to_try = None
        self.search_step = 0.5
        self.search_range = 5
        self.cen_range = None
        self.num_iter = 1
        self.algorithm = "gridrec"
        self.filter = "parzen"
        self.metadata = {}
        self.plotter = BqImPlotter_Center()
        self.plotter.create_app()
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
        self.find_center_button = Button(
            description="Click to automatically find center (image entropy).",
            disabled=False,
            button_style="info",
            tooltip="",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.index_to_try_textbox = IntText(
            description="Slice to use for auto:",
            disabled=False,
            style=extend_description_style,
            placeholder="Default is 1/2*y pixels",
        )
        self.num_iter_textbox = FloatText(
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
        self.find_center_vo_button = Button(
            description="Click to automatically find center (Vo).",
            disabled=False,
            button_style="info",
            tooltip="Vo's method",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )
        self.find_center_manual_button = Button(
            description="Click to find center by plotting.",
            disabled=False,
            button_style="info",
            tooltip="Start center-finding reconstruction with this button.",
            icon="",
            layout=Layout(width="auto", justify_content="center"),
        )

    def _center_update(self, change):
        self.current_center = change.new
        self.set_metadata()

    def _center_guess_update(self, change):
        self.center_guess = change.new
        self.set_metadata()

    def _load_rough_center_onclick(self, change):
        self.center_guess = self.Import.projections.pxX / 2
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

    def _center_textbox_slider_update(self, change):
        self.center_textbox.value = self.cen_range[change.new]
        self.current_center = self.center_textbox.value
        self.set_metadata()

    def find_center_on_click(self, change):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center`. Takes index_to_try and center_guess.
        This method has worked better for me, if I use a good index_to_try
        and center_guess.
        """
        prj_imgs = self.Import.projections.prj_imgs
        angles_rad = self.Import.projections.angles_rad

        self.find_center_button.button_style = "info"
        self.find_center_button.icon = "fa-spin fa-cog fa-lg"
        self.find_center_button.description = "Importing data..."
        if self.Import.projections.imported is True:
            prj_imgs = self.Import.projections.prj_imgs
            angles_rad = self.Import.projections.angles_rad
            self.Import.log.info("Finding center...")
            self.Import.log.info(f"Using index: {self.index_to_try}")
            self.find_center_button.description = "Finding center..."
            self.find_center_button.button_style = "info"
            self.current_center = find_center(
                prj_imgs,
                angles_rad,
                ratio=0.9,
                ind=self.index_to_try,
                init=self.center_guess,
            )
            self.center_textbox.value = self.current_center
            self.Import.log.info(f"Found center. {self.current_center}")
            self.find_center_button.description = "Found center."
            self.find_center_button.icon = "fa-check-square"
            self.find_center_button.button_style = "success"
        else:
            self.find_center_button.description = "Please import some data first."
            self.find_center_button.icon = "exclamation-triangle"
            self.find_center_button.button_style = "warning"

    def find_center_vo_on_click(self, change):
        """
        Callback to button for attempting to find center automatically using
        `tomopy.recon.rotation.find_center_vo`. Note: this method has not worked
        well for me.
        """
        prj_imgs = self.Import.projections.prj_imgs
        angles_rad = self.Import.projections.angles_rad
        self.find_center_vo_button.button_style = "info"
        self.find_center_vo_button.icon = "fa-spin fa-cog fa-lg"
        self.find_center_vo_button.description = "Importing data..."
        if self.Import.projections.imported is True:
            self.Import.log.info("Finding center using Vo method...")
            self.Import.log.info(f"Using index: {self.index_to_try}")
            self.find_center_vo_button.description = "Finding center using Vo method..."
            self.find_center_vo_button.button_style = "info"
            self.current_center = find_center_vo(prj_imgs, ncore=1)
            self.center_textbox.value = self.current_center
            self.Import.log.info(f"Found center. {self.current_center}")
            self.find_center_vo_button.description = "Found center."
            self.find_center_vo_button.icon = "fa-check-square"
            self.find_center_vo_button.button_style = "success"
        else:
            self.find_center_vo_button.description = "Please import some data first."
            self.find_center_vo_button.button_style = "warning"
            self.find_center_vo_button.icon = "exclamation-triangle"

    def find_center_manual_on_click(self, change):
        """
        Reconstructs at various centers when you click the button, and plots
        the results with a slider so one can view. TODO: see X example.
        Uses search_range, search_step, center_guess.
        Creates a :doc:`hyperslicer <mpl-interactions:examples/hyperslicer>` +
        :doc:`histogram <mpl-interactions:examples/hist>` plot
        """
        self.find_center_manual_button.button_style = "info"
        self.find_center_manual_button.icon = "fas fa-cog fa-spin fa-lg"
        self.find_center_manual_button.description = "Starting reconstruction."

        # TODO: for memory, add only desired slice
        prj_imgs = self.Import.projections.prj_imgs
        angles_rad = self.Import.projections.angles_rad
        cen_range = [
            self.center_guess - self.search_range,
            self.center_guess + self.search_range,
            self.search_step,
        ]

        # reconstruct, but also pull the centers used out to map to center
        # textbox
        self.rec, self.cen_range = write_center(
            prj_imgs,
            angles_rad,
            cen_range=cen_range,
            ind=self.index_to_try,
            mask=True,
            algorithm=self.algorithm,
            filter_name=self.filter,
            num_iter=self.num_iter,
        )
        if self.rec is None:
            self.find_center_manual_button.button_style = "warning"
            self.find_center_manual_button.icon = ""
            self.find_center_manual_button.description = (
                "Your projections do not have associated theta values."
            )

        self.plotter.plot(self.rec, self.Import.projections.filedir)
        self.find_center_manual_button.button_style = "success"
        self.find_center_manual_button.icon = "fa-check-square"
        self.find_center_manual_button.description = "Finished reconstruction."

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
        self.find_center_button.on_click(self.find_center_on_click)
        self.find_center_vo_button.on_click(self.find_center_vo_on_click)
        self.find_center_manual_button.on_click(self.find_center_manual_on_click)
        # Callback for index going to center
        self.plotter.image_index_slider.observe(
            self._center_textbox_slider_update, names="value"
        )

    def make_tab(self):
        """
        Function to create a Center object's :doc:`Tab <ipywidgets:index>`.
        """

        # Accordion to find center automatically
        self.automatic_center_vbox = VBox(
            [
                HBox(
                    [self.find_center_button, self.find_center_vo_button],
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

        # Accordion to find center manually
        self.manual_center_vbox = VBox(
            [
                HBox(
                    [self.find_center_manual_button],
                    layout=Layout(justify_content="center"),
                ),
                HBox(
                    [
                        VBox(
                            [
                                self.center_guess_textbox,
                                self.index_to_try_textbox,
                                self.num_iter_textbox,
                                self.search_range_textbox,
                                self.search_step_textbox,
                                self.algorithms_dropdown,
                                self.filters_dropdown,
                            ],
                            layout=Layout(
                                # display="flex",
                                # flex_flow="row wrap",
                                align_content="center",
                                justify_content="flex-start",
                            ),
                        ),
                        self.plotter.app,
                    ],
                    layout=Layout(justify_content="center"),
                ),
            ],
        )

        self.manual_center_accordion = Accordion(
            children=[self.manual_center_vbox],
            selected_index=None,
            titles=("Find center through plotting",),
        )

        self.tab = VBox(
            [
                HBox(
                    [self.center_textbox, self.load_rough_center],
                    layout=Layout(justify_content="center"),
                ),
                self.automatic_center_accordion,
                self.manual_center_accordion,
            ]
        )
