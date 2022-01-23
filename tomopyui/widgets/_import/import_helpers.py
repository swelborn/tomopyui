from ipywidgets import *
import functools

extend_description_style = {"description_width": "auto"}


def create_angles_textboxes(Import):
    """
    Creates textboxes for angle start/angle end. Currently, the No. Images
    textbox does do anything (tiff import grabs that number automatically).
    TODO: remove that.
    """

    def create_textbox(description, value, metadatakey, int=False):
        def angle_callbacks(change, key):
            Import.metadata[key] = change.new
            if key == "angle_start":
                Import.angle_start = Import.metadata[key]
            if key == "angle_end":
                Import.angle_end = Import.metadata[key]

        if int:
            textbox = IntText(
                value=value,
                description=description,
                disabled=False,
                style=extend_description_style,
            )

        else:
            textbox = FloatText(
                value=value,
                description=description,
                disabled=False,
                style=extend_description_style,
            )

        textbox.observe(
            functools.partial(angle_callbacks, key=metadatakey),
            names="value",
        )
        return textbox

    angle_start = create_textbox("Starting angle (\u00b0): ", -90, "angle_start")
    angle_end = create_textbox("Ending angle (\u00b0): ", 90, "angle_end")

    angles_textboxes = [angle_start, angle_end]
    return angles_textboxes
