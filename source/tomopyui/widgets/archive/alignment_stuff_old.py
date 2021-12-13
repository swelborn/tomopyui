        # tool for loading the range from above.


        load_range_from_above = Button(
            description="Click to load projection range from plot tab.",
            disabled=True,
            button_style="info",
            tooltip="Make sure to choose all of the buttons above before clicking this button",
            icon="",
            layout=Layout(width="95%", justify_content="center"),
        )


def radio_align_full_partial(change):
            if change.new == 1:
                projection_range_x_alignment.disabled = False
                projection_range_y_alignment.disabled = False
                load_range_from_above.disabled = False
                self.metadata["aligndata"] = True
                load_range_from_above.description = (
                    "Click to load projection range from above."
                )
                load_range_from_above.icon = ""
            elif change.new == 0:
                if "range_y_link" in locals() or "range_y_link" in globals():
                    range_y_link.unlink()
                    range_x_link.unlink()
                    load_range_from_above.button_style = "info"
                    load_range_from_above.description = (
                        "Unlinked ranges. Enable partial range to link again."
                    )
                    load_range_from_above.icon = "unlink"
                projection_range_x_alignment.value = [0, tomo.prj_imgs.shape[2] - 1]
                projection_range_x_alignment.disabled = True
                projection_range_y_alignment.value = [0, tomo.prj_imgs.shape[1] - 1]
                projection_range_y_alignment.disabled = True
                load_range_from_above.disabled = True
                self.metadata["aligndata"] = False