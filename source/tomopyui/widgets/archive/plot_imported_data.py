from ipywidgets import *
from matplotlib import animation
from matplotlib import pyplot as plt
from IPython.display import HTML
import plotly.express as px
from skimage.transform import rescale


def plot_imported_data(tomodata, widget_linker):

    extend_description_style = {"description_width": "auto"}
    plot_output = Output()
    movie_output = Output()

    def plot_projections(tomodata, range_x, range_y, range_z, skip, scale_factor):
        volume = tomodata.prj_imgs[
            range_z[0] : range_z[1] : skip,
            range_y[0] : range_y[1] : 1,
            range_x[0] : range_x[1] : 1,
        ].copy()
        volume_rescaled = rescale(
            volume, (1, scale_factor, scale_factor), anti_aliasing=False
        )
        fig = px.imshow(
            volume_rescaled,
            facet_col=0,
            facet_col_wrap=5,
            binary_string=True,
            height=2000,
            facet_row_spacing=0.001,
        )
        rangez = range(range_z[0], range_z[1], skip)
        angles = np.around(tomodata.theta * 180 / np.pi, decimals=1)
        for i, j in enumerate(rangez):
            for k in range(len(rangez)):
                if fig.layout.annotations[k]["text"] == "facet_col=" + str(i):
                    fig.layout.annotations[k]["text"] = (
                        "Proj:" + " " + str(j) + "<br>Angle:" + "" + str(angles[j])
                    )
                    fig.layout.annotations[k]["y"] = (
                        fig.layout.annotations[k]["y"] - 0.02
                    )
                    break
        fig.update_layout(
            # margin=dict(autoexpand=False),
            font_family="Helvetica",
            font_size=30,
            # margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor="LightSteelBlue",
        )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        # fig.for_each_annotation(lambda a: a.update(text=''))
        display(fig)

    def plot_projection_movie(tomodata, range_x, range_y, range_z, skip, scale_factor):

        frames = []
        animSliceNos = range(range_z[0], range_z[1], skip)
        volume = tomodata.prj_imgs[
            range_z[0] : range_z[1] : skip,
            range_y[0] : range_y[1] : 1,
            range_x[0] : range_x[1] : 1,
        ]
        volume_rescaled = rescale(
            volume, (1, scale_factor, scale_factor), anti_aliasing=False
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(animSliceNos)):
            frames.append([ax.imshow(volume_rescaled[i], cmap="viridis")])
        ani = animation.ArtistAnimation(
            fig, frames, interval=50, blit=True, repeat_delay=100
        )
        # plt.close()
        display(HTML(ani.to_jshtml()))

    def update_projection_plot_on_click(button_click):
        plot_output.clear_output()
        with plot_output:
            update_plot_button.button_style = "info"
            update_plot_button.icon = "fas fa-cog fa-spin fa-lg"
            update_plot_button.description = "Making a plot."
            plot_projections(
                tomodata,
                projection_range_x.value,
                projection_range_y.value,
                projection_range_theta.value,
                skip_theta.value,
                0.1,
            )
            update_plot_button.button_style = "success"
            update_plot_button.icon = "square-check"
            update_plot_button.description = "Do it again?"

    def create_projection_movie_on_click(button_click):
        movie_output.clear_output()
        with movie_output:
            create_movie_button.button_style = "info"
            create_movie_button.icon = "fas fa-cog fa-spin fa-lg"
            create_movie_button.description = "Making a movie."
            plot_projection_movie(
                tomodata,
                projection_range_x.value,
                projection_range_y.value,
                projection_range_theta.value,
                skip_theta.value,
                0.1,
            )
            create_movie_button.button_style = "success"
            create_movie_button.icon = "square-check"
            create_movie_button.description = "Do it again?"

    projection_range_x = IntRangeSlider(
        value=[0, tomodata.prj_imgs.shape[2] - 1],
        min=0,
        max=tomodata.prj_imgs.shape[2] - 1,
        step=1,
        description="Projection X Range:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )

    projection_range_y = IntRangeSlider(
        value=[0, tomodata.prj_imgs.shape[1] - 1],
        min=0,
        max=tomodata.prj_imgs.shape[1] - 1,
        step=1,
        description="Projection Y Range:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )

    projection_range_theta = IntRangeSlider(
        value=[0, tomodata.prj_imgs.shape[0] - 1],
        min=0,
        max=tomodata.prj_imgs.shape[0] - 1,
        step=1,
        description="Projection Z Range:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )

    skip_theta = IntSlider(
        value=20,
        min=1,
        max=50,
        step=1,
        description="Skipped range in z:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )

    create_movie_button = Button(
        description="Click me to create a movie", layout=Layout(width="auto")
    )
    create_movie_button.on_click(create_projection_movie_on_click)
    update_plot_button = Button(
        description="Click me to create a plot", layout=Layout(width="auto")
    )
    update_plot_button.on_click(update_projection_plot_on_click)

    movie_output.layout = Layout(width="100%", height="100%", align_items="center")
    plot_output.layout = Layout(width="100%", height="100%", align_items="center")
    plot_box_layout = Layout(
        border="3px solid blue",
        width="100%",
        height="auto",
        align_items="center",
        justify_content="center",
    )
    grid_plot = GridBox(
        children=[update_plot_button, plot_output],
        layout=Layout(
            width="100%",
            grid_template_rows="auto",
            grid_template_columns="15% 84%",
            grid_template_areas="""
            "update_plot_button plot_output"
            """,
        ),
    )
    grid_movie = GridBox(
        children=[create_movie_button, movie_output],
        layout=Layout(
            width="100%",
            grid_template_rows="auto",
            grid_template_columns="15% 84%",
            grid_template_areas="""
            "create_movie_button movie_output"
            """,
        ),
    )

    plot_vbox = VBox(
        [
            projection_range_x,
            projection_range_y,
            projection_range_theta,
            skip_theta,
            grid_movie,
            grid_plot,
        ],
        layout=plot_box_layout,
    )

    widget_linker["projection_range_x_movie"] = projection_range_x
    widget_linker["projection_range_y_movie"] = projection_range_y
    widget_linker["projection_range_theta_movie"] = projection_range_theta
    widget_linker["skip_theta_movie"] = skip_theta

    return plot_vbox
