import ipywidgets as widgets

# Generate a dummy list
Allfileslist = ["{}".format(x) for x in range(600)]
# Search box + generate some checboxes
search_widget = widgets.Text(
    placeholder="Type for older experiments", description="Search:", value=""
)
experiments = {}
options_widget = widgets.VBox(layout={"overflow": "auto"})
default_options = [
    widgets.Checkbox(description=eachfilename, value=False)
    for eachfilename in Allfileslist[-10:]
]


def whentextischanged(change):
    """Dynamically update the widget experiments"""
    search_input = change["new"]
    if search_input == "":
        # Reset search field, default to last 9 experiments
        new_options = default_options
    else:
        # Filter by search
        close_matches = [x for x in Allfileslist if search_input.lower() in x.lower()][
            :10
        ]
        for name in close_matches:
            if name not in experiments:
                experiments[name] = widgets.Checkbox(description=name, value=False)
        new_options = [experiments[eachfilename] for eachfilename in close_matches]

    options_widget.children = new_options


# Generate the vbox, search
multi_select = widgets.VBox([search_widget, options_widget])
search_widget.observe(whentextischanged, names="value")
multi_select
