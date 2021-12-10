from ipywidgets import *
from ipyfilechooser import FileChooser
import os
import shutil
import numpy as np

# TODO:
# overwrite rather than delete


def save_file_location(importmetadata, generalmetadata, tomodata):
    extend_description_style = {"description_width": "auto"}
    radio_save_drive = RadioButtons(
        options=["C:", "Z:"],
        description="Choose the drive you want to save the data on:",
        style=extend_description_style,
        layout=Layout(width="80%"),
    )

    def update_save_drive(change):
        if change.new == 1:
            workingdirectoryparent.reset(path="Z:/")
        elif change.new == 0:
            workingdirectoryparent.reset(path="C:/")

    radio_save_drive.observe(update_save_drive, names="index")

    # File chooser for the parent of the chosen working directory.
    if "fpath" not in importmetadata["tomo"]:
        workingdirectoryparent = FileChooser(path=r"C:/", show_only_dirs=True)
    else:
        workingdirectoryparent = FileChooser(
            path=importmetadata["tomo"]["fpath"], show_only_dirs=True
        )

    def update_wd(self):
        generalmetadata["workingdirectoryparentpath"] = self.selected_path

    workingdirectoryparent.register_callback(update_wd)

    workingdirectoryname = Text(
        value=(generalmetadata["analysis_date"] + "-analysis"),
        placeholder="recon",
        description="Working Directory Name:",
        disabled=False,
        style=extend_description_style,
    )

    def overwrite_button_on_click(self):
        self.icon = "fa-cog"
        os.chdir(workingdirectoryparent.selected_path)
        try:
            shutil.rmtree(workingdirectoryname.value)
            self.icon = "fa-check-square"
            self.description = (
                "Directory deleted. Click button above to make a new directory."
            )
            self.button_style = "success"
        except:
            print("Unsuccessful directory removal")

    overwrite_button = Button(
        description="",
        disabled=True,
        button_style="",
        tooltip="",
        layout=Layout(width="99%"),
    )

    mkdir_button = Button(
        description="Make the directory above",
        disabled=False,
        button_style="",
        tooltip="Make the directory.",
        layout=Layout(width="99%"),
    )

    def mkdir_on_button_click(self):
        os.chdir(workingdirectoryparent.selected_path)
        if os.path.isdir(workingdirectoryname.value):
            self.button_style = "warning"
            self.description = "Directory already exists"
            self.icon = "fa-exclamation-triangle"
            overwrite_button.description = "Click to delete. Otherwise, rename above."
            overwrite_button.disabled = False
            overwrite_button.button_style = "danger"
            overwrite_button.tooltip = "Warning: this will overwrite all data "
            overwrite_button.icon = "question"
        else:
            self.button_style = "success"
            self.description = "Directory Created"
            self.icon = "fa-check-square"
            save_tomo_data_button.disabled = False
            save_tomo_data_button.description = (
                "Click to save normalized tomograms in your working directory."
            )
            os.mkdir(workingdirectoryname.value)
            generalmetadata["workingdirectoryname"] = workingdirectoryname.value
            generalmetadata["workingdirectorypath"] = (
                generalmetadata["workingdirectoryparentpath"]
                + "\\"
                + workingdirectoryname.value
            )

    mkdir_button.on_click(mkdir_on_button_click)
    overwrite_button.on_click(overwrite_button_on_click)

    recon_box_layout = Layout(
        border="3px solid blue",
        width="50%",
        height="auto",
        align_items="center",
        justify_content="center",
    )

    working_directory_parent_filechooser_hb = HBox(
        [Label(value="Working Directory Parent: "), workingdirectoryparent]
    )

    def save_tomo_data_on_click(self):
        os.chdir(generalmetadata["workingdirectorypath"])
        self.description = "Saving"
        self.icon = "gear"
        np.save("tomo_norm_mlog", tomodata.prj_imgs)
        self.description = "Done."
        self.icon = "fa-square-check"
        self.button_style = "success"

    save_tomo_data_button = Button(
        description="",
        disabled=True,
        button_style="info",
        tooltip="Save raw tomodata in the working directory you just made.",
        layout=Layout(width="99%"),
    )
    save_tomo_data_button.on_click(save_tomo_data_on_click)

    save_data_box = VBox(
        children=[
            radio_save_drive,
            working_directory_parent_filechooser_hb,
            workingdirectoryname,
            mkdir_button,
            overwrite_button,
            save_tomo_data_button,
        ],
        layout=recon_box_layout,
    )

    return save_data_box
