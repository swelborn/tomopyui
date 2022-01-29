# TODO: reimplement this
class DataExplorerTab:
    def __init__(
        self,
        Align,
        Recon,
    ):
        self.align_de = DataExplorer(Align)
        self.recon_de = DataExplorer(Recon)
        self.fb_de = DataExplorer()

    def create_data_explorer_tab(self):
        self.recent_alignment_accordion = Accordion(
            children=[self.align_de.data_plotter],
            selected_index=None,
            titles=("Plot Recent Alignments",),
        )
        self.recent_recon_accordion = Accordion(
            children=[self.recon_de.data_plotter],
            selected_index=None,
            titles=("Plot Recent Reconstructions",),
        )
        self.analysis_browser_accordion = Accordion(
            children=[self.fb_de.data_plotter],
            selected_index=None,
            titles=("Plot Any Analysis",),
        )

        self.tab = VBox(
            children=[
                self.analysis_browser_accordion,
                self.recent_alignment_accordion,
                self.recent_recon_accordion,
            ]
        )


class DataExplorer:
    def __init__(
        self, obj: (Align or Recon) = None, single_image=False, imagestacks=None
    ):
        self.figs = None
        self.single_image = single_image
        self.images = None
        self.scales = None
        self.projection_num_sliders = None
        self.imagestacks_metadata = None
        self.plays = None
        self.imagestacks = [np.zeros((15, 100, 100)) for i in range(2)]
        self.linked_stacks = False
        self.obj = obj
        self._init_widgets()

    def _init_widgets(self):
        self.button_style = {"font_size": "22px"}
        self.button_layout = Layout(width="45px", height="40px")
        self.Plotter = Plotter()
        self.app_output = Output()
        if self.obj is not None:
            if self.obj.widget_type == "Align":
                self.run_list_selector = Select(
                    options=[],
                    rows=5,
                    description="Alignments:",
                    disabled=False,
                    style=extend_description_style,
                    layout=Layout(justify_content="center"),
                )
                self.linked_stacks = True
                self.titles = ["Before Alignment", "After Alignment"]
                self.load_run_list_button = Button(
                    description="Load alignment list",
                    icon="download",
                    button_style="info",
                    layout=Layout(width="auto"),
                )
            else:
                self.run_list_selector = Select(
                    options=[],
                    rows=5,
                    description="Reconstructions:",
                    disabled=False,
                    style=extend_description_style,
                    layout=Layout(justify_content="center"),
                )
                self.linked_stacks = False
                self.titles = ["Projections", "Reconstruction"]
                self.load_run_list_button = Button(
                    description="Load reconstruction list",
                    icon="download",
                    button_style="info",
                    layout=Layout(width="auto"),
                )

            self.run_list_selector.observe(self.choose_file_to_plot, names="value")
            self.load_run_list_button.on_click(self._load_run_list_on_click)
            self._create_plotter_run_list()

        elif self.single_image:
            self.titles = ["Projections"]
        else:

            self.titles = ["Projections", "Reconstruction"]
            self.filebrowser = Filebrowser()
            self.filebrowser.create_file_browser()
            self.filebrowser.load_data_button.on_click(self.load_data_from_filebrowser)
            self._create_plotter_filebrowser()

    def load_data_from_filebrowser(self, change):
        metadata = {}
        metadata["filedir"] = self.filebrowser.metadata["parent_filedir"]
        metadata["filename"] = self.filebrowser.metadata["parent_filename"]
        metadata["angle_start"] = self.filebrowser.metadata["angle_start"]
        metadata["angle_end"] = self.filebrowser.metadata["angle_end"]
        tomo = TomoData(metadata=metadata)
        self.imagestacks[0] = tomo.prj_imgs
        metadata["filedir"] = str(self.filebrowser.selected_method)
        metadata["filename"] = str(self.filebrowser.selected_data_filename)
        tomo = TomoData(metadata=metadata)
        # TODO: make this agnostic to recon/tomo
        self.imagestacks[1] = tomo.prj_imgs
        if self.filebrowser.selected_analysis_type == "recon":
            self.titles = ["Projections", "Reconstruction"]
        else:
            self.titles = ["Before Alignment", "After Alignment"]
        self.create_figures_and_widgets()
        self._create_image_app()

    def find_file_in_metadata(self, foldername):
        for run in range(len(self.obj.run_list)):
            if foldername in self.obj.run_list[run]:
                metadata = {}
                metadata["filedir"] = self.obj.run_list[run][foldername][
                    "parent_filedir"
                ]
                metadata["filename"] = self.obj.run_list[run][foldername][
                    "parent_filename"
                ]
                metadata["angle_start"] = self.obj.run_list[run][foldername][
                    "angle_start"
                ]
                metadata["angle_end"] = self.obj.run_list[run][foldername]["angle_end"]
                self.imagestacks[0] = TomoData(metadata=metadata).prj_imgs
                metadata["filedir"] = self.obj.run_list[run][foldername]["savedir"]
                if self.obj.widget_type == "Align":
                    metadata["filename"] = "projections_after_alignment.tif"
                else:
                    metadata["filename"] = "recon.tif"
                self.imagestacks[1] = TomoData(metadata=metadata).prj_imgs
                self.create_figures_and_widgets()
                self._create_image_app()

    def create_figures_and_widgets(self):
        if self.single_image:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_one_plot(self.imagestacks, self.titles)
        elif self.linked_stacks:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_two_plots_with_single_slider(
                self.imagestacks, self.titles
            )
            self.projection_num_sliders = [self.projection_num_sliders]
            self.plays = [self.plays]

        else:
            (
                self.figs,
                self.images,
                self.scales,
                self.projection_num_sliders,
                self.plays,
            ) = self.Plotter._create_two_plots_with_two_sliders(
                self.imagestacks, self.titles
            )

        self.vmin_vmax_sliders = self._create_vmin_vmax_sliders()
        self.remove_high_low_intensity_buttons = (
            self._create_remove_high_low_intensity_buttons()
        )
        self.swapaxes_buttons = self._create_swapaxes_buttons()
        self.reset_button = Button(
            icon="redo", style=self.button_style, layout=self.button_layout
        )
        self.reset_button.on_click(self._reset_on_click)

        self.reset_button_one_fig = Button(
            icon="redo", style=self.button_style, layout=self.button_layout
        )
        self.reset_button_one_fig.on_click(self._reset_on_click_one_fig)

    def _create_vmin_vmax_sliders(self):
        vmin = self.imagestacks[0].min()
        vmax = self.imagestacks[0].max()
        slider1 = FloatRangeSlider(
            description="vmin-vmax:",
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 1000,
            value=(vmin, vmax),
            orientation="vertical",
        )

        def change_vmin_vmax1(change):
            self.scales[0]["image"].min = change["new"][0]
            self.scales[0]["image"].max = change["new"][1]

        slider1.observe(change_vmin_vmax1, names="value")

        vmin = self.imagestacks[1].min()
        vmax = self.imagestacks[1].max()
        slider2 = FloatRangeSlider(
            description="vmin-vmax:",
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 1000,
            value=(vmin, vmax),
            orientation="vertical",
        )

        def change_vmin_vmax2(change):
            self.scales[1]["image"].min = change["new"][0]
            self.scales[1]["image"].max = change["new"][1]

        slider2.observe(change_vmin_vmax2, names="value")

        sliders = [slider1, slider2]

        return sliders

    def _create_swapaxes_buttons(self):
        def swapaxes_on_click1(change):
            # defaults to going with the high/low value from
            self.imagestacks[0] = self.Plotter._swap_axes_on_click(
                self.imagestacks[0],
                self.images[0],
                self.projection_num_sliders[0],
            )

        def swapaxes_on_click2(change):
            # defaults to going with the high/low value from
            self.imagestacks[1] = self.Plotter._swap_axes_on_click(
                self.imagestacks[1],
                self.images[1],
                self.projection_num_sliders[1],
            )

        button1 = Button(
            icon="random", layout=self.button_layout, style=self.button_style
        )
        # button1.button_style = "info"
        button2 = Button(
            icon="random", layout=self.button_layout, style=self.button_style
        )
        # button2.button_style = "info"
        button1.on_click(swapaxes_on_click1)
        button2.on_click(swapaxes_on_click2)
        buttons = [button1, button2]
        return buttons

    def _create_remove_high_low_intensity_buttons(self):
        """
        Parameters
        ----------
        imagestack: np.ndarray
            images that it will use to find vmin, vmax - this is found by
            getting the 0.5 and 99.5 percentiles of the data
        scale: dict

        """

        def remove_high_low_intensity_on_click1(change):
            # defaults to going with the high/low value from
            self.Plotter._remove_high_low_intensity_on_click(
                self.imagestacks[0], self.scales[0], self.vmin_vmax_sliders[0]
            )

        def remove_high_low_intensity_on_click2(change):
            # defaults to going with the high/low value from
            self.Plotter._remove_high_low_intensity_on_click(
                self.imagestacks[1], self.scales[1], self.vmin_vmax_sliders[1]
            )

        button1 = Button(
            icon="adjust", layout=self.button_layout, style=self.button_style
        )
        button1.button_style = "info"
        button2 = Button(
            icon="adjust", layout=self.button_layout, style=self.button_style
        )
        button2.button_style = "info"
        button1.on_click(remove_high_low_intensity_on_click1)
        button2.on_click(remove_high_low_intensity_on_click2)
        buttons = [button1, button2]
        return buttons

    def choose_file_to_plot(self, change):
        self.find_file_in_metadata(change.new)

    def _load_run_list_on_click(self, change):
        self.load_run_list_button.button_style = "info"
        self.load_run_list_button.icon = "fas fa-cog fa-spin fa-lg"
        self.load_run_list_button.description = "Importing run list."
        # creates a list from the keys in pythonic way
        # from https://stackoverflow.com/questions/11399384/extract-all-keys-from-a-list-of-dictionaries
        # don't know how it works
        self.run_list_selector.options = list(
            set().union(*(d.keys() for d in self.obj.run_list))
        )
        self.load_run_list_button.button_style = "success"
        self.load_run_list_button.icon = "fa-check-square"
        self.load_run_list_button.description = "Finished importing run list."

    def _reset_on_click(self, change):
        self.create_figures_and_widgets()
        self._create_image_app()

    def _reset_on_click_one_fig(self, change):
        self.create_figures_and_widgets()
        self._create_image_app_raw_import()

    def _create_image_app(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        footer_layout = Layout(justify_content="center")
        header = None

        self.button_box1 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[0],
                self.swapaxes_buttons[0],
            ],
            layout=left_sidebar_layout,
        )
        self.button_box2 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[1],
                self.swapaxes_buttons[1],
            ],
            layout=right_sidebar_layout,
        )

        left_sidebar = VBox(
            [self.vmin_vmax_sliders[0], self.button_box1], layout=left_sidebar_layout
        )
        center = HBox(self.figs, layout=Layout(justify_content="center"))
        right_sidebar = VBox(
            [self.vmin_vmax_sliders[1], self.button_box2], layout=right_sidebar_layout
        )
        if self.linked_stacks:
            footer = HBox(
                self.plays + self.projection_num_sliders, layout=footer_layout
            )
        else:
            footer = HBox(
                [
                    HBox([self.plays[0], self.projection_num_sliders[0]]),
                    HBox([self.plays[1], self.projection_num_sliders[1]]),
                ],
                layout=footer_layout,
            )
        self.image_app = AppLayout(
            header=header,
            left_sidebar=left_sidebar,
            center=center,
            right_sidebar=right_sidebar,
            footer=footer,
            pane_widths=[0.5, 5, 0.5],
            pane_heights=[0, 10, "40px"],
            height="auto",
        )
        with self.app_output:
            self.app_output.clear_output(wait=True)
            display(self.image_app)

    def _create_image_app_raw_import(self):
        left_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        right_sidebar_layout = Layout(
            justify_content="space-around", align_items="center"
        )
        footer_layout = Layout(justify_content="center")
        header = None

        self.button_box1 = VBox(
            [
                self.reset_button,
                self.remove_high_low_intensity_buttons[0],
                self.swapaxes_buttons[0],
            ],
            layout=left_sidebar_layout,
        )

        left_sidebar = VBox(
            [self.vmin_vmax_sliders[0], self.button_box1], layout=left_sidebar_layout
        )
        center = HBox(self.figs, layout=Layout(justify_content="center"))
        right_sidebar = None

        if self.linked_stacks:
            footer = HBox(
                self.plays + self.projection_num_sliders, layout=footer_layout
            )

        self.image_app = AppLayout(
            header=header,
            left_sidebar=left_sidebar,
            center=center,
            right_sidebar=right_sidebar,
            footer=footer,
            pane_widths=[0.5, 5, 0.5],
            pane_heights=[0, 10, "40px"],
            height="auto",
        )
        with self.app_output:
            self.app_output.clear_output(wait=True)
            display(self.image_app)

    def _create_plotter_run_list(self):
        # self.create_figures_and_widgets()
        # self._create_image_app()
        self.data_plotter = VBox(
            [self.load_run_list_button, self.run_list_selector, self.app_output]
        )

    def _create_plotter_filebrowser(self):
        # self.create_figures_and_widgets()
        # self._create_image_app()
        self.data_plotter = VBox([self.filebrowser.filebrowser, self.app_output])


class Filebrowser:
    def __init__(self):

        # parent directory filechooser
        self.orig_data_fc = FileChooser()
        self.orig_data_fc.register_callback(self.update_orig_data_folder)
        self.fc_label = Label("Original Data", layout=Layout(justify_content="Center"))

        # subdirectory selector
        self.subdir_list = []
        self.subdir_label = Label(
            "Analysis Directories", layout=Layout(justify_content="Center")
        )
        self.subdir_selector = Select(options=self.subdir_list, rows=5, disabled=False)
        self.subdir_selector.observe(self.populate_methods_list, names="value")
        self.selected_subdir = None

        # method selector
        self.methods_label = Label("Methods", layout=Layout(justify_content="Center"))
        self.methods_list = []
        self.methods_selector = Select(
            options=self.methods_list, rows=5, disabled=False
        )
        self.methods_selector.observe(self.populate_data_list, names="value")
        self.selected_method = None

        # data selector
        self.data_label = Label("Data", layout=Layout(justify_content="Center"))
        self.data_list = []
        self.data_selector = Select(options=self.data_list, rows=5, disabled=False)
        self.data_selector.observe(self.set_data_filename, names="value")
        self.allowed_extensions = (".npy", ".tif", ".tiff")
        self.selected_data_filename = None
        self.selected_data_ftype = None
        self.selected_analysis_type = None

        self.options_metadata_table_output = Output()

        # load data button
        self.load_data_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="info",
            layout=Layout(width="75px", height="86px"),
        )

    def populate_subdirs_list(self):
        self.subdir_list = [
            pathlib.PurePath(f) for f in os.scandir(self.root_filedir) if f.is_dir()
        ]
        self.subdir_list = [
            subdir.parts[-1]
            for subdir in self.subdir_list
            if any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.subdir_selector.options = self.subdir_list

    def update_orig_data_folder(self):
        self.root_filedir = self.orig_data_fc.selected_path
        self.populate_subdirs_list()
        self.methods_selector.options = []

    def populate_methods_list(self, change):
        self.selected_subdir = pathlib.PurePath(self.root_filedir) / change.new
        self.methods_list = [
            pathlib.PurePath(f) for f in os.scandir(self.selected_subdir) if f.is_dir()
        ]
        self.methods_list = [
            subdir.parts[-1]
            for subdir in self.methods_list
            if not any(x in subdir.parts[-1] for x in ("-align", "-recon"))
        ]
        self.methods_selector.options = self.methods_list

    def populate_data_list(self, change):
        if change.new is not None:
            self.selected_method = (
                pathlib.PurePath(self.root_filedir) / self.selected_subdir / change.new
            )
            self.file_list = [
                pathlib.PurePath(f)
                for f in os.scandir(self.selected_method)
                if not f.is_dir()
            ]
            self.data_list = [
                file.name
                for file in self.file_list
                if any(x in file.name for x in self.allowed_extensions)
            ]
            self.data_selector.options = self.data_list
            self.load_metadata()
        else:
            self.data_selector.options = []

    def set_data_filename(self, change):
        self.selected_data_filename = change.new
        self.selected_data_ftype = pathlib.PurePath(self.selected_data_filename).suffix
        if "recon" in pathlib.PurePath(self.selected_subdir).name:
            self.selected_analysis_type = "recon"
        elif "align" in pathlib.PurePath(self.selected_subdir).name:
            self.selected_analysis_type = "align"

    def load_metadata(self):
        self.metadata_file = [
            self.selected_method / file.name
            for file in self.file_list
            if "metadata.json" in file.name
        ]
        if self.metadata_file != []:
            self.metadata = load_metadata(fullpath=self.metadata_file[0])
            self.options_table = metadata_to_DataFrame(self.metadata)
            with self.options_metadata_table_output:
                self.options_metadata_table_output.clear_output(wait=True)
                display(self.options_table)

    def create_file_browser(self):
        fc = VBox([self.fc_label, self.orig_data_fc])
        subdir = VBox([self.subdir_label, self.subdir_selector])
        methods = VBox([self.methods_label, self.methods_selector])
        data = VBox([self.data_label, self.data_selector])
        button = VBox(
            [
                Label("Upload", layout=Layout(justify_content="center")),
                self.load_data_button,
            ]
        )
        top_hb = HBox(
            [fc, subdir, methods, data, button],
            layout=Layout(justify_content="center"),
            align_items="stretch",
        )
        box = VBox(
            [top_hb, self.options_metadata_table_output],
            layout=Layout(justify_content="center", align_items="center"),
        )
        self.filebrowser = box
