import tomopyui.widgets.meta as meta
from ipywidgets import *

a = meta.Import()

import_widgets = [a.filechooser, a.angles_textboxes, a.opts_checkboxes]
import_box = HBox([import_widgets])
