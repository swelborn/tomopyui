from ipywidgets import *
from ipyfilechooser import FileChooser
from skimage.transform import rescale
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from tomopy.widgets.file_chooser_recon import file_chooser_recon
import tomopy.data.tomodata as td


def plot_aligned_data(
    reconmetadata,
    alignmentmetadata,
    importmetadata,
    generalmetadata,
    alignmentdata,
    widget_linker,
):
    # Importing file chooser box
    recon_files, uploaders, opts_for_uploaders, angles_hb = file_chooser_recon(
        reconmetadata, generalmetadata
    )
    widget_linker["recon_files"] = recon_files

    # Initialize sliders, etc.
    extend_description_style = {"description_width": "auto"}
    skip_theta = IntSlider()
    projection_range_x = IntRangeSlider(
        description="Projection X Range:",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )
    projection_range_y = IntRangeSlider(
        description="Projection Y Range:",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )
    projection_range_theta = IntRangeSlider(
        description="Projection Theta Range:",
        layout=Layout(width="70%"),
        style=extend_description_style,
    )

    def load_data_for_plot(self):

        projection_range_x.description = "Projection X Range:"
        projection_range_y.description = "Projection Y Range:"
        projection_range_theta.description = "Projection Theta Range:"
        skip_theta.description = "Skip theta:"
        projection_range_x.max = current_tomo.prj_imgs.shape[2] - 1
        projection_range_y.max = current_tomo.prj_imgs.shape[1] - 1
        projection_range_theta.max = current_tomo.prj_imgs.shape[0] - 1
        projection_range_x.value = [0, current_tomo.prj_imgs.shape[2] - 1]
        projection_range_y.value = [0, current_tomo.prj_imgs.shape[1] - 1]
        projection_range_theta.value = [0, current_tomo.prj_imgs.shape[0] - 1]

        projection_range_x.min = 0
        projection_range_x.step = 1
        projection_range_x.disabled = False
        projection_range_x.continuous_update = False
        projection_range_x.orientation = "horizontal"
        projection_range_x.readout = True
        projection_range_x.readout_format = "d"
        projection_range_x.layout = Layout(width="70%")
        projection_range_x.style = extend_description_style

        projection_range_y.min = 0
        projection_range_y.step = 1
        projection_range_y.disabled = False
        projection_range_y.continuous_update = False
        projection_range_y.orientation = "horizontal"
        projection_range_y.readout = True
        projection_range_y.readout_format = "d"
        projection_range_y.layout = Layout(width="70%")
        projection_range_y.style = extend_description_style

        projection_range_theta.min = 0
        projection_range_theta.step = 1
        projection_range_theta.disabled = False
        projection_range_theta.continuous_update = False
        projection_range_theta.orientation = "horizontal"
        projection_range_theta.readout = True
        projection_range_theta.readout_format = "d"
        projection_range_theta.layout = Layout(width="70%")
        projection_range_theta.style = extend_description_style

        skip_theta.value = 20
        skip_theta.min = 1
        skip_theta.max = 50
        skip_theta.step = 1
        skip_theta.disabled = False
        skip_theta.continuous_update = False
        skip_theta.orientation = "horizontal"
        skip_theta.readout = True
        skip_theta.readout_format = "d"
        skip_theta.layout = Layout(width="70%")
        skip_theta.style = extend_description_style

    load_data_button = Button(
        description="Click to load the selected data.", layout=Layout(width="auto")
    )
    load_data_button.on_click(load_data_for_plot)

    # Radio for use of raw/normalized data, or normalized + aligned.

    def create_tomo_from_fc(dropdown_choice):
        tomo = td.TomoData(metadata=reconmetadata["tomo"]["tomo_0"])
        for key in reconmetadata["tomo"]:
            print(key)
            if "fname" in reconmetadata["tomo"][key]:
                print(reconmetadata["tomo"][key]["fname"])
                if reconmetadata["tomo"][key]["fname"] == dropdown_choice:
                    tomo = td.TomoData(metadata=reconmetadata["tomo"][key])
                    print(tomo.prj_imgs)
        return tomo

    def define_tomo_dropdown_options(alignmentdata, uploaders):
        aligned_dropdown_options = []
        uploaded_dropdown_options = []
        num_uploaders = len(uploaders)
        if uploaders is not None:
            active_tomo_fc_list = [
                uploaders[i]
                for i in range(num_uploaders)
                if uploaders[i].selected_path is not None
            ]
            uploaded_dropdown_options = [
                active_tomo_fc_list[i].selected_filename
                for i in range(len(active_tomo_fc_list))
            ]
        if alignmentdata is not None:
            aligned_tomo_dropdown_options = [
                f"alignment_{i}" for i in range(len(alignmentdata))
            ]
        tomo_dropdown.options = (
            ["..."] + uploaded_dropdown_options + aligned_dropdown_options
        )
        tomo_dropdown.value = "..."

    def upload_current_tomo(self):
        global current_tomo
        if self["new"].__contains__("alignment_"):
            alignment_number = int(filter(str.isdigit, self["new"]))
            current_tomo = alignmentdata[alignment_number].tomo
        else:
            print(self["new"])
            current_tomo = create_tomo_from_fc(self["new"])
            print(current_tomo.prj_imgs)

    tomo_dropdown = Dropdown(
        options=["...", "Upload data..."],
        value="...",
        description="Normalized Tomo:",
        disabled=False,
        style=extend_description_style,
    )

    define_tomo_dropdown_options(alignmentdata, uploaders)
    tomo_dropdown.observe(upload_current_tomo, names="value")

    def update_dropdownmenu(self):
        key = self.title
        reconmetadata["tomo"][key]["fpath"] = self.selected_path
        reconmetadata["tomo"][key]["fname"] = self.selected_filename
        define_tomo_dropdown_options(alignmentdata, uploaders)

    uploader_no = 10
    for i in range(uploader_no):
        uploaders[i].register_callback(update_dropdownmenu)

    plot_output = Output()
    movie_output = Output()

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

    def create_projection_movie_on_click(button_click):
        movie_output.clear_output()
        with movie_output:
            create_movie_button.button_style = "info"
            create_movie_button.icon = "fas fa-cog fa-spin fa-lg"
            create_movie_button.description = "Making a movie."
            plot_projection_movie(
                current_tomo,
                projection_range_x.value,
                projection_range_y.value,
                projection_range_theta.value,
                skip_theta.value,
                0.1,
            )
            create_movie_button.button_style = "success"
            create_movie_button.icon = "square-check"
            create_movie_button.description = "Do it again?"

    # Making a movie button
    create_movie_button = Button(
        description="Click me to create a movie", layout=Layout(width="auto")
    )
    create_movie_button.on_click(create_projection_movie_on_click)
    movie_output = Output()
    movie_output.layout = Layout(width="100%", height="100%", align_items="center")

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

    plot_box_layout = Layout(
        border="3px solid blue",
        width="100%",
        height="auto",
        align_items="center",
        justify_content="center",
    )

    plot_vbox = VBox(
        [
            tomo_dropdown,
            load_data_button,
            projection_range_x,
            projection_range_y,
            projection_range_theta,
            skip_theta,
            grid_movie,
        ],
        layout=plot_box_layout,
    )

    return plot_vbox, recon_files

    # widget_linker["projection_range_x_movie"] = projection_range_x
    # widget_linker["projection_range_y_movie"] = projection_range_y
    # widget_linker["projection_range_theta_movie"] = projection_range_theta
    # widget_linker["skip_theta_movie"] = skip_theta


## USE LATER FOR RECONSTRUCTION PLOTTING:

# if aligned_or_uploaded_radio.value == "Last Recon:":
#     projection_range_x.description = "Recon X Range:"
#     projection_range_y.description = "Recon Y Range:"
#     projection_range_Z.description = "Recon Z Range:"
#     skip_theta.description = "Skip z:"
#     projection_range_x.max = current_tomo.shape[2] - 1
#     projection_range_x.value = [0, current_tomo.shape[2] - 1]
#     projection_range_y.max = current_tomo.shape[1] - 1
#     projection_range_y.value = [0, current_tomo.shape[1] - 1]
#     projection_range_theta.value = [0, current_tomo.shape[0] - 1]
#     projection_range_theta.max = current_tomo.shape[0] - 1
