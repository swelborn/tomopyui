from ipyfilechooser import FileChooser
from ipywidgets import *
import functools


# will create one file uploader, given a specific working directory and a title
# for the uploader


def file_chooser_recon(reconmetadata):
    extend_description_style = {"description_width": "auto"}
    uploader_no = 10
    uploaders = [FileChooser(path=cwd, title=f"tomo_{i}") for i in range(uploader_no)]

    # def update_fnames(self):
    #     key = self.title
    #     reconmetadata["tomo"][key]["fpath"] = self.selected_path
    #     reconmetadata["tomo"][key]["fname"] = self.selected_filename

    reconmetadata["tomo"] = {f"tomo_{i}": {} for i in range(uploader_no)}
    # for i in range(uploader_no):
    #     uploaders[i].register_callback(update_fnames)

    ############### Creating options checkboxes
    def create_option_dictionary(opt_list):
        opt_dictionary = {opt.description: opt.value for opt in opt_list}
        return opt_dictionary

    def create_dict_on_checkmark_import(change, opt_list, dictname):
        reconmetadata["tomo"][dictname]["opts"] = create_option_dictionary(opt_list)

    def create_import_option_checkbox(description, disabled=False, value=0):
        checkbox = Checkbox(description=description, disabled=disabled, value=value)
        return checkbox

    other_import_options = [
        "rotate",
    ]
    # this will be in the same order as uploaders
    # (uploaders[i] matches opts_for_uploaders[i])
    opts_for_uploaders = [[] for n in range(uploader_no)]
    i = 0
    for key in reconmetadata["tomo"]:
        for opt in other_import_options:
            opts_for_uploaders[i].append(create_import_option_checkbox(opt))
        # make them clickable, creates dictionary when clicked
        [
            opt.observe(
                functools.partial(
                    create_dict_on_checkmark_import, opt_list=[opt], dictname=key,
                ),
                names=["value"],
            )
            for opt in opts_for_uploaders[i]
        ]
        i += 1

    # Similar to above, we create angle start/end textboxes for all:

    def angle_callbacks(change, description, tomo_number):
        reconmetadata["tomo"][f"tomo_{tomo_number}"][description] = change.new

    def create_angles_textboxes(tomo_number):
        angle_start_textbox = FloatText(
            value=-90,
            description="Starting angle (\u00b0):",
            disabled=False,
            style=extend_description_style,
        )

        angle_end_textbox = FloatText(
            value=89.5,
            description="Ending angle (\u00b0):",
            disabled=False,
            style=extend_description_style,
        )

        # currently unused. automatically grabs num_theta from files
        number_of_projections_textbox = IntText(
            value=360,
            description="Number of Images",
            disabled=False,
            style=extend_description_style,
        )

        reconmetadata["tomo"][f"tomo_{tomo_number}"][
            "start_angle"
        ] = angle_start_textbox.value
        reconmetadata["tomo"][f"tomo_{tomo_number}"][
            "end_angle"
        ] = angle_end_textbox.value

        angle_start_textbox.observe(
            functools.partial(
                angle_callbacks, description="start_angle", tomo_number=tomo_number
            ),
            names="value",
        )
        angle_end_textbox.observe(
            functools.partial(
                angle_callbacks, description="end_angle", tomo_number=tomo_number
            ),
            names="value",
        )

        angles_hbox = HBox(
            [angle_start_textbox, angle_end_textbox, number_of_projections_textbox]
        )

        return angles_hbox

    angles_hb = [
        create_angles_textboxes(tomo_number) for tomo_number in range(len(uploaders))
    ]

    ####### Combining everything
    uploaders_textboxes_checkboxes = [
        HBox([uploaders[i], angles_hb[i], *opts_for_uploaders[i]])
        for i in range(len(uploaders))
    ]
    recon_files = VBox(
        uploaders_textboxes_checkboxes,
        layout=Layout(flex_flow="row wrap", width="100%"),
    )
    return recon_files, uploaders, opts_for_uploaders, angles_hb
