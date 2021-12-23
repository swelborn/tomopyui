from ipywidgets import *
from ipyfilechooser import FileChooser
import functools
import tomopy.data.tomodata as td
import pdb

default_generalmetadata = {"analysis_date": "20000101"}
default_importmetadata = dict(tomo={}, flat={}, dark={})


def create_import_box(
    importmetadata=default_importmetadata, generalmetadata=default_generalmetadata
):

    cwd = generalmetadata["starting_wd"]
    main_logger = generalmetadata["main_logger"]
    main_handler = generalmetadata["main_handler"]

    extend_description_style = {"description_width": "auto"}
    radio_drive_import = RadioButtons(
        options=["C:", "Z:"],
        description="Choose the drive your data is on:",
        style=extend_description_style,
    )

    def update_drive(change):
        if change.new == 1:
            tomofc.reset(path="Z:/")
            darkfc.reset(path="Z:/")
            flatfc.reset(path="Z:/")
        elif change.new == 0:
            tomofc.reset(path="C:/")
            darkfc.reset(path="C:/")
            flatfc.reset(path="C:/")

    # make sure the file choosers are going to the correct directory.
    radio_drive_import.observe(update_drive, names="index")

    # File choosers for each type of data
    tomofc = FileChooser(path=cwd)
    darkfc = FileChooser(path=cwd)
    flatfc = FileChooser(path=cwd)

    # defining importmetadata callbacks. You should select tomo first to define the file path.
    def update_tomofname(self):
        importmetadata["tomo"]["fpath"] = self.selected_path
        importmetadata["tomo"]["fname"] = self.selected_filename
        darkfc.reset(path=self.selected_path)
        flatfc.reset(path=self.selected_path)

    def update_flatfname(self):
        importmetadata["flat"]["fpath"] = self.selected_path
        importmetadata["flat"]["fname"] = self.selected_filename

    def update_darkfname(self):
        importmetadata["dark"]["fpath"] = self.selected_path
        importmetadata["dark"]["fname"] = self.selected_filename

    tomofc.register_callback(update_tomofname)
    flatfc.register_callback(update_flatfname)
    darkfc.register_callback(update_darkfname)

    def update_datatype(self):
        if self["owner"].description == "Tomo Image Type:":
            importmetadata["tomo"]["imgtype"] = self["new"]
        if self["owner"].description == "Flat Image Type:":
            importmetadata["flat"]["imgtype"] = self["new"]
        if self["owner"].description == "Dark Image Type:":
            importmetadata["dark"]["imgtype"] = self["new"]

    radio_import_options = ["tiff", "tiff folder", "h5", "one image"]

    def create_filetype_radio(
        description, options=radio_import_options, value="tiff", disabled=False
    ):
        radio = RadioButtons(
            options=options,
            value=value,
            description=description,
            disabled=disabled,
            style={"description_width": "auto"},
        )
        return radio

    # create radio buttons for image type
    tomo_radio = create_filetype_radio("Tomo Image Type:")
    flat_radio = create_filetype_radio("Flat Image Type:")
    dark_radio = create_filetype_radio("Dark Image Type:")

    # make radio buttons do something
    tomo_radio.observe(update_datatype, names="value")
    flat_radio.observe(update_datatype, names="value")
    dark_radio.observe(update_datatype, names="value")

    # initialize metadata. probably can find a way to do this in the create_filetype_radio function.
    importmetadata["tomo"]["imgtype"] = "tiff"
    importmetadata["flat"]["imgtype"] = "tiff"
    importmetadata["dark"]["imgtype"] = "tiff"

    for key in importmetadata:
        importmetadata[key]["imgtype"] = "tiff"
        importmetadata[key]["opts"] = {}
        importmetadata[key]["fpath"] = None
        importmetadata[key]["fname"] = None

    def create_option_dictionary(opt_list):
        opt_dictionary = {opt.description: opt.value for opt in opt_list}
        return opt_dictionary

    def create_dict_on_checkmark_import(change, opt_list, dictname):
        importmetadata[dictname]["opts"] = create_option_dictionary(opt_list)

    # create checkboxes for other import options
    def create_import_option_checkbox(description, disabled=False, value=0):
        checkbox = Checkbox(description=description, disabled=disabled, value=value)
        return checkbox

    # create other import options
    other_import_options = ["downsample", "rotate", "jawn1", "jawn2"]
    for key in importmetadata:
        importmetadata[key]["opts"] = {opt: False for opt in other_import_options}

    tomo_import_other_options = []
    flat_import_other_options = []
    dark_import_other_options = []
    for key in importmetadata:
        if key == "tomo":
            for opt in other_import_options:
                tomo_import_other_options.append(create_import_option_checkbox(opt))
            # make them clickable, creates dictionary when clicked
            [
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark_import,
                        opt_list=tomo_import_other_options,
                        dictname=key,
                    ),
                    names=["value"],
                )
                for opt in tomo_import_other_options
            ]
        if key == "flat":
            for opt in other_import_options:
                flat_import_other_options.append(create_import_option_checkbox(opt))
            # make them clickable, creates dictionary when clicked
            [
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark_import,
                        opt_list=flat_import_other_options,
                        dictname=key,
                    ),
                    names=["value"],
                )
                for opt in flat_import_other_options
            ]
        if key == "dark":
            for opt in other_import_options:
                dark_import_other_options.append(create_import_option_checkbox(opt))
            # make them clickable, creates dictionary when clicked
            [
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark_import,
                        opt_list=dark_import_other_options,
                        dictname=key,
                    ),
                    names=["value"],
                )
                for opt in dark_import_other_options
            ]

    # function to create grid of checkboxes
    def assign_checkbox_to_grid(optlist, grid_size_horiz=2, grid_size_vert=2):
        _optlist = optlist
        grid = GridspecLayout(grid_size_vert, grid_size_horiz)
        grid[0, 0] = _optlist[0]
        grid[0, 1] = _optlist[1]
        grid[1, 0] = _optlist[2]
        grid[1, 1] = _optlist[3]
        return grid

    # create grid of checkboxes
    tomo_import_other_options = assign_checkbox_to_grid(tomo_import_other_options)
    flat_import_other_options = assign_checkbox_to_grid(flat_import_other_options)
    dark_import_other_options = assign_checkbox_to_grid(dark_import_other_options)

    # create upload button
    def parse_upload_type(metadata, datadict):
        for key in metadata:
            if metadata[key]["fpath"] is not None:
                datadict[key] = td.TomoData(metadata=metadata[key])

        return datadict

    def upload_data_on_click(self):
        if self.button_style == "success" or isinstance("tomo_norm_mlog", td.TomoData):
            self.button_style = "warning"
            self.description = "It seems you already uploaded your data. Upload again?"
            self.icon = "exclamation-triangle"
        elif self.icon == "question":
            self.button_style = ""
            self.icon = ""
            self.description = "Press this button to upload data into memory."
            self.tooltip = "Upload your datasets (tomo, dark, and flat chosen above)"
        elif self.button_style == "" or self.button_style == "warning":
            self.button_style = "info"
            self.icon = "fas fa-cog fa-spin fa-lg"
            self.description = "Uploading data."
            try:
                datadict = {}
                importmetadata["tomo"]["start_angle"] = angle_start_textbox.value
                importmetadata["tomo"]["end_angle"] = angle_end_textbox.value
                importmetadata["tomo"][
                    "num_theta"
                ] = number_of_projections_textbox.value
                datadict = parse_upload_type(importmetadata, datadict)
                self.button_style = "success"
                self.description = "Upload complete."
                self.icon = "fa-check-square"
                if "flat" in datadict and "dark" in datadict:
                    tomo_norm, importmetadata["tomo_norm_mlog"] = td.normalize(
                        datadict["tomo"], datadict["flat"], datadict["dark"]
                    )
                    main_logger.info("Normalized the data.")
                else:
                    importmetadata["tomo_norm_mlog"] = datadict["tomo"]
                    main_logger.info(
                        "Darks and flats have not been uploaded into memory. Assuming your data is already normalized and -log."
                    )
            except:
                self.icon = "question"
                self.description = r"That didn't work. Sure you chose the correct files/formats? Click again to reset."
                self.button_style = "warning"

    upload_data_button = Button(
        description="Press this button to upload data into memory.",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Upload your datasets (tomo, dark, and flat chosen above)",
        layout=Layout(width="auto"),
    )
    upload_data_button.on_click(upload_data_on_click)

    # create container for all functions above
    raw_import_box_layout = Layout(
        border="3px solid blue",
        width="90%",
        align_items="center",
        justify_content="center",
        padding="50px",
    )
    inner_box_layout = Layout(
        border="2px solid green",
        width="100%",
        align_items="stretch",
        margin="10px 0px 10px 0px",
    )

    tomo_import_hb = HBox(
        [
            Label(value="Tomo data", layout=Layout(width="100px")),
            tomofc,
            tomo_radio,
            tomo_import_other_options,
        ],
        layout=inner_box_layout,
    )
    flat_import_hb = HBox(
        [
            Label(value="Flat data", layout=Layout(width="100px")),
            flatfc,
            flat_radio,
            flat_import_other_options,
        ],
        layout=inner_box_layout,
    )
    dark_import_hb = HBox(
        [
            Label(value="Dark data", layout=Layout(width="100px")),
            darkfc,
            dark_radio,
            dark_import_other_options,
        ],
        layout=inner_box_layout,
    )

    text_description_style = {"description_width": "auto"}

    angle_start_textbox = FloatText(
        value=-90,
        description="Starting angle (\u00b0):",
        disabled=False,
        style=text_description_style,
    )

    angle_end_textbox = FloatText(
        value=89.5,
        description="Ending angle (\u00b0):",
        disabled=False,
        style=text_description_style,
    )

    number_of_projections_textbox = IntText(
        value=360,
        description="Number of Images",
        disabled=False,
        style=text_description_style,
    )

    angles_hb = HBox(
        [
            radio_drive_import,
            angle_start_textbox,
            angle_end_textbox,
            number_of_projections_textbox,
        ]
    )

    raw_data_import_box = VBox(
        children=[
            angles_hb,
            tomo_import_hb,
            flat_import_hb,
            dark_import_hb,
            upload_data_button,
        ],
        layout=raw_import_box_layout,
    )

    return (
        generalmetadata,
        importmetadata,
        raw_data_import_box,
    )
