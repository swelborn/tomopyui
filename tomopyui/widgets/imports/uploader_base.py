import pathlib
from abc import ABC, abstractmethod

from ipyfilechooser import FileChooser
from tomopyui.backend.helpers import file_finder
from tomopyui.widgets.styles import (
    extend_description_style,
)
from tomopyui.backend.io import ProjectionsBase
from tomopyui.widgets.helpers import ImportButton
from tomopyui.widgets.view import BqImViewer_Projections_Parent
from IPython.display import display
from typing import List, Optional
import ipywidgets as widgets
import solara



class UploaderBase(ABC):
    """"""

    def __init__(self):
        self.files_not_found_str: str = ""
        self.filetypes_to_look_for: List[str] = []
        self.imported_metadata: bool = False

        self.projections: Optional[ProjectionsBase] = None

        # File browser
        self.filechooser: FileChooser = FileChooser()
        self.filechooser.register_callback(self._update_quicksearch_from_filechooser)
        self.filedir: Optional[pathlib.Path] = None
        self.filename: Optional[str] = None

        # Quick path search textbox
        self.quick_path_search = widgets.Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=widgets.Layout(align_items="stretch"),
        )
        self.quick_path_search.observe(
            self._update_filechooser_from_quicksearch, names="value"
        )

        # Import button, disabled before you put anything into the quick path
        # see helpers class
        self.import_button = ImportButton(self.import_data)

        # Where metadata will be displayed
        self.metadata_table_output = widgets.Output()

        # Progress bar showing upload progress
        self.progress_output = widgets.Output()

        # Save tiff checkbox
        self.save_tiff_on_import_checkbox = widgets.Checkbox(
            description="Save .tif on import.",
            value=False,
            style=extend_description_style,
            disabled=False,
        )

        # Create data visualizer
        self.viewer = BqImViewer_Projections_Parent()
        self.viewer.create_app()

        # Will update based on the import status
        self.import_status_label = widgets.Label(
            layout=widgets.Layout(justify_content="center")
        )

        # Will update when searching for metadata
        self.find_metadata_status_label = widgets.Label(
            layout=widgets.Layout(justify_content="center")
        )

    def _update_filechooser_from_quicksearch(self, change):
        """
        Updates the filechooser based on input from the quick search textbox. This method
        checks if the path exists, updates the file directory for strings in
        self.filetypes_to_look_for, and then runs the subclass-specific function
        self.update_filechooser_from_quicksearch.

        Parameters
        ----------
        change: The change notification from the quick search textbox. change.new contains
                the new string value from the textbox.
        """
        path = pathlib.Path(change.new.strip())
        self.import_button.disable()
        self.imported_metadata = False

        # with self.metadata_table_output:
        #     self.metadata_table_output.clear_output(wait=True)
        #     display(self.find_metadata_status_label)

        # Determine if the path is a directory or file and set accordingly
        if not path.exists():
            self.find_metadata_status_label.value = (
                "No file or directory with that name."
            )
            return

        # Update filechooser only if the new path or filename differs from the current selection
        new_path = str(path.parent) if path.is_file() else str(path)
        new_filename = path.name if path.is_file() else None

        if (self.filechooser.selected_path != new_path) or (
            self.filechooser.selected_filename != new_filename
        ):
            self.filechooser.reset(path=new_path, filename=new_filename)
            self.filedir = pathlib.Path(new_path)
            self.filename = new_filename

        # Find files in the directory that match the filetypes to look for
        found_files = (
            file_finder(self.filedir, self.filetypes_to_look_for)
            if self.filedir
            else []
        )
        if not found_files:
            filetype_str = " or ".join(self.filetypes_to_look_for)
            self.find_metadata_status_label.value = (
                f"No {filetype_str} files found in this directory."
            )
            self.files_found = False
        else:
            self.files_found = True
            self.update_filechooser_from_quicksearch(found_files)

    def _update_quicksearch_from_filechooser(self, filechooser: Optional[FileChooser]):
        """
        Updates the quick search box after selection from the file chooser. This
        triggers self._update_filechooser_from_quicksearch(), so not much logic is
        needed other than setting the filedirectory and filename.
        """
        if not filechooser:
            return

        sel_path = filechooser.selected_path
        sel_filename = filechooser.selected_filename
        if sel_path:
            self.filedir = pathlib.Path(sel_path)
            self.filename = sel_filename if sel_filename else None
            full_filepath = (
                str(self.filedir / self.filename)
                if self.filename
                else str(self.filedir)
            )
            if full_filepath != self.quick_path_search.value:
                self.quick_path_search.value = full_filepath

        # changing value of quick_path_search manually may not work...
        # if self.filedir:
        #     full_filepath = str(self.filedir / self.filename) if self.filename else str(self.filedir)
        #     if full_filepath == self.quick_path_search.value:
        #         _ = DummyChange(full_filepath)
        #         self._update_filechooser_from_quicksearch(_)
        #     else:
        #         self.quick_path_search.value = full_filepath

    # Each uploader has a method to update the filechooser from the quick search path,
    # and vice versa.
    @abstractmethod
    def update_filechooser_from_quicksearch(self, change): ...

    # Each uploader has a method to import data given the filepath chosen in the
    # filechooser/quicksearch box
    @abstractmethod
    def import_data(self): ...


# class DummyChange:
#     def __init__(self, new):
#         self.new = new
