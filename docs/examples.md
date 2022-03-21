# Tutorial
:::{note}

This is a tutorial as of v0.0.4.

:::
## Opening Jupyter Lab

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/intro.mp4", autoplay=False)
```

## General Tutorial (latest)
```{jupyter-execute}
:hide-code:

# add these options if video looks weird: width=600, height=300
from IPython.display import YouTubeVideo
YouTubeVideo('O2RJCL4x4JE')
```

## Importing Data

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/import.mp4", autoplay=False)
```

Once you import your data, you can check it out and save a movie of it:
```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/looking_at_data_fast.mp4", autoplay=False)
```

## Finding Center of Rotation

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/center.mp4", autoplay=False)
```

## Alignment (with CUDA)

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/alignment.mp4", autoplay=False)
```

This alignment creates subfolders with metadata:

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/subdirs.mp4", autoplay=False)
```

## Exploring aligned data

```{jupyter-execute}
:hide-code:

from ipywidgets import Video, HBox, Layout
Video.from_file("_static/videos/data_explorer_fast.mp4", autoplay=False)
```