## This is experimental

import pathlib

from ipywidgets import *

from tomopyui.backend.hdf_handler import HDF5_Handler
from tomopyui.backend.io import Projections_Prenormalized
from tomopyui.widgets.hdf_viewer import *
from tomopyui.widgets.helpers import ToggleIconButton
from tomopyui.widgets.imports import UploaderBase
from tomopyui.widgets.styles import button_font_style


class HDF5_GeneralUploader(UploaderBase):
    def __init__(self, filedir: pathlib.Path = None):
        #
        super().__init__()
        self.files_not_found_str = ""
        self.filetypes_to_look_for = [".hdf5", ".h5"]
        del self.save_tiff_on_import_checkbox
        self.projections = Projections_Prenormalized()
        self.viewer = BqImViewer_HDF5()
        self.viewer.create_app()
        self.hdf_handler = HDF5_Handler(self)
        self.viewer.hdf_handler = self.hdf_handler

        # Filechooser stuff
        if filedir is not None:
            self.initial_filedir = filedir
            self.filechooser.reset(path=filedir)
        self.filechooser.show_only_dirs = True
        self.filechooser.dir_icon_append = True

        # Initialize widgets/observes
        self.init_widgets()
        self.init_observes()

    def init_state(self):
        self.turn_off_callbacks = True
        self.hdf_handler.close()
        self.files_sel.options = []
        self.files_sel.value = None
        self.turn_off_callbacks = False

    def init_widgets(self):
        self.button_layout = Layout(width="45px", height="40px")
        self.files_sel = Select()
        # Go faster on play button
        self.home_button: ipywidgets.Button = Button(
            icon="home",
            layout=self.button_layout,
            style=button_font_style,
            tooltip="Speed up play slider.",
        )
        self.reset_button: ipywidgets.Button = Button(
            icon="redo",
            layout=self.button_layout,
            style=button_font_style,
            tooltip="Reset to original.",
        )
        self.tree_output = Output()
        self.load_ds_checkbox = Checkbox(
            value=False,
            description="Load 8x Downsampled Data",
            disabled=False,
            indent=False,
        )

    def init_observes(self):
        self.files_sel.observe(self.init_populate_groups, names="value")
        self.reset_button.on_click(self.reset_button_cb)

    def init_populate_groups(self, *args):
        if self.turn_off_callbacks:
            return
        if self.filedir is None:
            return
        self.filepath = self.filedir / self.files_sel.value
        self.hdf_handler.close()
        self.hdf_handler.new_tree(self.filepath)
        with self.tree_output:
            display(self.hdf_handler.tree.widget)

    def reset_button_cb(self, *args):
        self.init_state()

    def import_data(self):
        pass

    def _update_filechooser_from_quicksearch(self, change):
        self.init_state()
        super()._update_filechooser_from_quicksearch(change)

    def update_filechooser_from_quicksearch(self, files):
        self.files_sel.options = files
        if len(files) == 1:
            self.files_sel.value = self.files_sel.options[0]

    def create_app(self):
        self.app = HBox(
            [
                VBox(
                    [
                        HBox(
                            [
                                self.reset_button,
                                self.filechooser,
                                self.load_ds_checkbox,
                            ]
                        ),
                        HBox(
                            [
                                self.quick_path_search,
                                self.files_sel,
                            ]
                        ),
                        self.tree_output,
                    ],
                ),
                self.viewer.app,
            ],
            layout=Layout(justify_content="center"),
        )


class HDF5_MultipleEnergyUploader(HDF5_GeneralUploader):
    def __init__(self):
        super().__init__()
        self.projections1 = self.projections
        self.projections2 = Projections_Prenormalized()
        self.viewer1 = BqImViewer_HDF5_Align_To()
        self.viewer1.create_app()
        self.viewer1.projections = self.projections1
        self.viewer2 = BqImViewer_HDF5_Align(self.viewer1)
        self.viewer2.create_app()
        self.viewer2.projections = self.projections2
        self.toggle_button = ToggleIconButton(self.viewer2_on, self.viewer1_on)
        self.viewer = self.viewer1
        self.hdf_handler.viewer = self.viewer
        self.viewer1.hdf_handler = self.hdf_handler
        self.viewer2.hdf_handler = self.hdf_handler
        self.create_app()

    def viewer1_on(self):
        self.viewer = self.viewer1
        self.hdf_handler.viewer = self.viewer
        self.hdf_handler.projections = self.projections1
        self.projections = self.projections1
        self.hdf_handler.ds_factor_from_parent = False

    def viewer2_on(self):
        self.viewer = self.viewer2
        self.hdf_handler.viewer = self.viewer
        self.hdf_handler.projections = self.projections2
        self.projections = self.projections2
        self.hdf_handler.ds_factor_from_parent = True
        self.hdf_handler.loaded_ds_factor = self.viewer1.ds_dropdown.value

    def create_app(self):
        self.app = HBox(
            [
                VBox(
                    [
                        HBox(
                            [
                                self.reset_button,
                                self.filechooser,
                                self.load_ds_checkbox,
                            ]
                        ),
                        HBox(
                            [
                                self.quick_path_search,
                                self.files_sel,
                            ]
                        ),
                        self.tree_output,
                    ],
                ),
            ],
            layout=Layout(justify_content="center"),
        )
