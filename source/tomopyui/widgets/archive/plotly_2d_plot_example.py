# From
import pandas as pd
import plotly.express as px


def handle_elem_change(change):
    with rangewidg.hold_trait_notifications():  # This is because if you do't put it it set max,

        rangewidg.max = big_grid[
            dropm_elem.value
        ].max()  # and if max is < min he freaks out. Like this he first
        rangewidg.min = big_grid[
            dropm_elem.value
        ].min()  # set everything and then send the eventual errors notification.
        rangewidg.value = [
            big_grid[dropm_elem.value].min(),
            big_grid[dropm_elem.value].max(),
        ]


def plot_change(change):
    df = big_grid[big_grid["id_col"].isin(dropm_id.value)]
    output.clear_output(wait=True)
    with output:
        fig = px.scatter(
            df,
            x="coord1",
            y="coord2",
            color=dropm_elem.value,
            hover_data=["info"],
            width=500,
            height=800,
            color_continuous_scale="Turbo",
            range_color=rangewidg.value,
        )
        fig.show()


# define the widgets dropm_elem and rangewidg, which are the possible df.columns and the color range
# used in the function plot.
big_grid = pd.DataFrame(
    data=dict(
        id_col=[1, 2, 3, 4, 5],
        col1=[0.1, 0.2, 0.3, 0.4, 0.5],
        col2=[10, 20, 30, 40, 50],
        coord1=[6, 7, 8, 9, 10],
        coord2=[6, 7, 8, 9, 10],
        info=[
            "info1",
            "info2",
            "info3",
            "info4",
            "info5",
        ],
    )
)
list_elem = ["col1", "col2", "info"]
list_id = big_grid.id_col.values


dropm_elem = widgets.Dropdown(
    options=list_elem
)  # creates a widget dropdown with all the _ppms
dropm_id = widgets.SelectMultiple(
    options=list_id, description="Active Jobs", disabled=False
)

rangewidg = widgets.FloatRangeSlider(
    value=[big_grid[dropm_elem.value].min(), big_grid[dropm_elem.value].max()],
    min=big_grid[dropm_elem.value].min(),
    max=big_grid[dropm_elem.value].max(),
    step=0.001,
    readout_format=".3f",
    description="Color Scale Range",
    continuous_update=False,
)
output = widgets.Output()
# this line is crucial, it basically says: Whenever you move the dropdown menu widget, call the function
# #handle_elem_change, which will in turn update the values of rangewidg
dropm_elem.observe(handle_elem_change, names="value")
dropm_elem.observe(plot_change, names="value")
dropm_id.observe(plot_change, names="value")
rangewidg.observe(plot_change, names="value")

# # #this line is also crucial, it links the widgets dropmenu and rangewidg with the function plot, assigning
# # #to elem and to rang (parameters of function plot) the values of dropmenu and rangewidg

left_box = widgets.VBox([output])
right_box = widgets.VBox([dropm_elem, rangewidg, dropm_id])
tbox = widgets.HBox([left_box, right_box])
# widgets.interact(plot,elem=dropm_elem,rang=rangewidg)

display(tbox)
