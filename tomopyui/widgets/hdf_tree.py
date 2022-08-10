import h5py
import os
import pathlib

from ipywidgets import *


class HDF5_Tree:

    # Lots of logic from hdf5object.py
    def __init__(self, hdf_obj, depth, hdf_handler):
        self.hdf_obj = hdf_obj
        self.children = {}
        self.children_widgets = {}
        self.datasets = []
        self.depth = depth
        self.vbox = []
        self.parent_group = self.hdf_obj.parent
        self.hdf_handler = hdf_handler
        if isinstance(self.hdf_obj, h5py.Group):
            self.group_name = self.hdf_obj.name

        for name, obj in self.hdf_obj.items():
            if isinstance(obj, h5py.Dataset):
                self.datasets.append(name)
            elif isinstance(obj, h5py.Group):
                self.children[name] = HDF5_Tree(obj, self.depth + 1, self.hdf_handler)
                self.children_widgets[name] = self.children[name].widget

        # create accordion object
        def group_selection_changed(change):
            if change["new"] is not None:
                name = change["new"]
                self.hdf_handler.selected_group_name = pathlib.PurePosixPath(
                    self.hdf_obj[self.groups_titles[name]].name
                )
                self.hdf_handler.selected_group = self.hdf_obj[self.groups_titles[name]]
                self.hdf_handler.load_data()
            else:
                if len(self.datasets) > 0:
                    self.datasets_select.value = None
                self.hdf_handler.selected_dataset_name = None
                self.hdf_handler.selected_dataset = None
                self.hdf_handler.selected_group_name = pathlib.PurePosixPath(
                    self.group_name
                )
                self.hdf_handler.selected_group = self.hdf_obj
                self.hdf_handler.load_data()

        def close_selected_dataset():
            self.datasets_select.value = None

        self.groups_accordions = Accordion(
            children=[val for val in self.children_widgets.values()],
            titles=tuple(title for title in self.children_widgets),
            selected_index=None,
        )
        self.groups_accordions.observe(group_selection_changed, names="selected_index")

        self.vbox.append(self.groups_accordions)

        # name each accordion sub-object
        self.groups_titles = []
        for val in self.children:
            self.groups_titles.append(val)

        def dataset_selection_changed(change):
            datasets_info = []
            dataset = change["new"]
            self.hdf_handler.selected_dataset_name = self.hdf_obj[dataset].name
            self.hdf_handler.selected_dataset = self.hdf_obj[dataset]
            values = [
                f"<b>{self.hdf_obj[dataset].name}</b>",
                f"<i>shape</i>: {self.hdf_obj[dataset].shape}",
                f"<i>dtype</i>: {self.hdf_obj[dataset].dtype}",
            ]
            for attr in self.hdf_obj[dataset].attrs:
                values.append(f"<i>{attr}</i>: {self.hdf_obj[dataset].attrs[attr]}")
            datasets_info.append(
                widgets.HTML(value="<br/>".join(values), layout=Layout(width="100%"))
            )
            self.datasets_info.children = datasets_info

        # init vbox for dataset attributes and info
        self.datasets_info = VBox([])

        # create Select widget for datasets
        if len(self.datasets) > 0:
            self.datasets_select = Select(options=self.datasets)
            self.datasets_select.observe(dataset_selection_changed, names="value")
            self.vbox.append(self.datasets_select)
            self.vbox.append(self.datasets_info)

        # create VBox for attributes
        attributes = []
        for attr in self.hdf_obj.attrs:
            attributes.append(f"<i>{attr}</i>: {self.hdf_obj.attrs[attr]}")
        self.attributes = widgets.HTML(
            value="<br/>".join(attributes), layout=Layout(width="100%")
        )

        # create widget
        if len(attributes) > 0:
            titles = ["Groups / Datasets", "Attributes"]
            self.widget = widgets.Tab([VBox(self.vbox), self.attributes], titles=titles)
        else:
            self.widget = VBox(self.vbox)
