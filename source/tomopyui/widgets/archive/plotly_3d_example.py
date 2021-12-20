# This is from https://plotly.com/python/visualizing-mri-volume-slices/


import time
import numpy as np

from skimage import io

# vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
# volume = vol.T
volume = tomo_norm_mlog.prj_imgs[0:100]
volume = tomopy.misc.morph.downsample(vol, level=2, axis=1)
volume = tomopy.misc.morph.downsample(vol, level=2, axis=2)
# volume = tomo_norm_mlog.prj_imgs[0:10]
r, c = volume[0].shape

# Define frames
import plotly.graph_objects as go

nb_frames = volume.shape[0]

fig = go.Figure(
    frames=[
        go.Frame(
            data=go.Surface(
                z=(6.7 - k * 0.1) * np.ones((r, c)),
                surfacecolor=np.flipud(volume[9 - k]),
                cmin=0,
                cmax=1,
            ),
            name=str(
                k
            ),  # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)
    ]
)

# Add data to be displayed before animation starts
fig.add_trace(
    go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[99]),
        colorscale="Gray",
        cmin=0,
        cmax=1,
        colorbar=dict(thickness=20, ticklen=4),
    )
)


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

# Layout
fig.update_layout(
    title="Slices in volumetric data",
    width=600,
    height=600,
    scene=dict(
        zaxis=dict(range=[-0.1, 6.8], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders,
)

fig.show()
