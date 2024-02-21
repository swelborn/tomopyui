from enum import Enum
import ipywidgets as widgets
from IPython.display import display
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Optional, Type, Callable, Any
from typing import get_origin, get_args
from pydantic import BaseModel


class WidgetGroupingConfig(BaseModel):
    groups: list[list[str]]


class WidgetStyleConfig(BaseModel):
    description_width: str = "initial"


class WidgetLayoutConfig(BaseModel):
    margin: str = "2px"
    width: Optional[str] = "25%"


class HBoxLayoutConfig(BaseModel):
    display: str = "flex"
    flex_flow: str = "row wrap"
    align_items: str = "flex-start"
    justify_content: str = "flex-start"


class MetadataInput:
    def __init__(
        self,
        model_cls: Type[BaseModel],
        grouping_config: Optional[WidgetGroupingConfig] = None,
        style_config: Optional[WidgetStyleConfig] = None,
        layout_config: Optional[WidgetLayoutConfig] = None,
        hbox_config: Optional[HBoxLayoutConfig] = None,
    ) -> None:
        self.model_cls = model_cls
        self.model = model_cls()  # Instance of the Pydantic model
        self.widgets = self.create_widgets_from_model()
        # Set defaults if configurations are not provided
        self.grouping_config = grouping_config or WidgetGroupingConfig(groups=[])
        self.style_config = style_config or WidgetStyleConfig()
        self.layout_config = layout_config or WidgetLayoutConfig()
        self.hbox_config = hbox_config or HBoxLayoutConfig()
        self.out = widgets.Output()
        self.create_metadata_widget_box()

    def create_widgets_from_model(self) -> dict[str, widgets.Widget]:
        widgets_dict: dict[str, widgets.Widget] = {}
        for field_name, field in self.model.model_fields.items():
            widget = self.create_widget_for_field(field_name, field)
            if widget:
                widget.observe(
                    self.create_on_value_change_callback(field_name), names="value"
                )
                widgets_dict[field_name] = widget
        return widgets_dict

    def create_widget_for_field(
        self, field_name: str, field_info: FieldInfo
    ) -> Optional[widgets.Widget]:
        """Create a widget based on the field type and metadata."""
        field_type, default_value = self.get_field_type_and_default(field_info)

        widget = None

        # Check if the field is an enum and create a dropdown if it is
        if issubclass(field_type, Enum):
            widget = self.create_enum_widget(
                field_type, default_value, field_info.description
            )
        else:
            widget = self.create_standard_widget(
                field_type, default_value, field_info.description
            )

        if widget is not None:
            widget.observe(
                self.create_on_value_change_callback(field_name), names="value"
            )

        return widget

    def create_enum_widget(
        self, enum_type: Type[Enum], default_value: Any, description: Optional[str]
    ) -> widgets.Widget:
        """Create a dropdown widget for Enum fields."""
        choices = list(enum_type)
        default_value = default_value if default_value in choices else choices[0]
        return widgets.Dropdown(
            options=[
                (item.name, item) for item in choices
            ],  # Use name for better readability
            value=default_value,
            description=description or "",
        )

    def create_standard_widget(
        self, field_type: Type[Any], default_value: Any, description: Optional[str]
    ) -> Optional[widgets.Widget]:
        """Create a standard widget for basic Python field types."""
        type_to_widget = {
            float: widgets.FloatText,
            int: widgets.IntText,
            str: widgets.Text,
        }
        widget_class = type_to_widget.get(field_type)
        if widget_class:
            return widget_class(value=default_value, description=description or "")
        else:
            # Log or handle the unsupported field_type as needed
            print(
                f"Warning: No widget found for field type {field_type}. Using Text widget as fallback."
            )
            return widgets.Text(value=str(default_value), description=description or "")

    def get_field_type_and_default(self, field_info: FieldInfo) -> tuple[Any, Any]:
        """Determine the Python type for a field and an appropriate default value."""
        base_annotation = get_origin(field_info.annotation) or field_info.annotation
        args_annotation = get_args(field_info.annotation)
        is_optional = type(None) in args_annotation
        field_type = (
            args_annotation[0]
            if is_optional and len(args_annotation) > 1
            else base_annotation
        )

        # Determine a sensible default value
        if field_type in [int, float]:
            default_value = 0
        else:
            default_value = ""
        return field_type, (
            field_info.default if field_info.default is not None else default_value
        )

    def create_on_value_change_callback(self, field_name: str) -> Callable:
        def on_value_change(change: dict[str, Any]) -> None:
            setattr(self.model, field_name, change["new"])

        return on_value_change

    def create_model(self) -> BaseModel:
        # Directly return the updated model instance
        return self.model

    def grouping_configuration(self) -> list[list[str]]:
        return self.grouping_config.groups

    def widget_style(self) -> dict[str, str]:
        return {"description_width": self.style_config.description_width}

    def widget_layout(self) -> widgets.Layout:
        return widgets.Layout(
            margin=self.layout_config.margin, width=self.layout_config.width
        )

    def hbox_layout(self) -> widgets.Layout:
        return widgets.Layout(
            display=self.hbox_config.display,
            flex_flow=self.hbox_config.flex_flow,
            align_items=self.hbox_config.align_items,
            justify_content=self.hbox_config.justify_content,
        )

    def create_widget_groups(self) -> list[widgets.Widget]:
        """Create groups of widgets based on the grouping configuration."""
        grouped_widgets: list[widgets.Widget] = [
            widgets.HTML(value="<h3>Set Metadata Here</h3>")
        ]

        # Check if there are specific groupings defined
        if self.grouping_configuration():
            for group in self.grouping_configuration():
                group_widgets = [
                    self.create_styled_widget(
                        self.widgets[field_name],
                        self.widget_style(),
                        self.widget_layout(),
                    )
                    for field_name in group
                    if field_name in self.widgets
                ]

                hbox = (
                    widgets.HBox(group_widgets, layout=self.hbox_layout())
                    if len(group_widgets) > 1
                    else group_widgets[0]
                )
                grouped_widgets.append(hbox)
        else:
            # If no groupings are defined, add widgets individually
            for _, widget in self.widgets.items():
                styled_widget = self.create_styled_widget(
                    widget, self.widget_style(), self.widget_layout()
                )
                grouped_widgets.append(styled_widget)

        return grouped_widgets

    def create_metadata_widget_box(self):
        """Create and display the metadata widget box with grouped widgets."""
        global_container_layout = widgets.Layout(
            width="100%"
        )  # Global container layout

        grouped_widgets = self.create_widget_groups()

        self.metadata_widget_box = widgets.VBox(
            children=grouped_widgets,
            layout=global_container_layout,
        )

    def create_styled_widget(
        self, widget: widgets.Widget, style: dict[str, str], layout: widgets.Layout
    ) -> widgets.Widget:
        """Apply style and layout to the widget and return it."""
        widget.set_trait("layout", layout)
        widget.set_trait("style", style)
        return widget

    def clear_output(self):
        self.out.clear_output()

    def display_widgets(self) -> None:
        """Override to display the metadata_widget_box instead of individual widgets."""
        with self.out:
            self.clear_output()
            display(self.metadata_widget_box)

    # def create_model(self) -> PrenormProjectionsMetadata:
    #     meta = {}
    #     for name, widget in self.required_metadata:
    #         meta[name] = widget.value

    #     return PrenormProjectionsMetadata(**meta)

    # def create_metadata_callback(self, name, widget):
    #     def callback(change):
    #         if not self.projections.metadata.imported:
    #             self.projections.metadata.metadata[name] = widget.value
    #         if all(
    #             param in self.projections.metadata.metadata
    #             for param in self.required_parameters
    #         ):
    #             self.create_and_display_metadata_tables()

    #     return callback

    # def enter_metadata_output(self):
    #     for name, val, widget in zip(
    #         self.required_parameters, self.init_required_values, self.widgets_to_enable
    #     ):
    #         if name not in self.projections.metadata.metadata:
    #             widget.disabled = False
    #             self.projections.metadata.metadata[name] = val
    #         else:
    #             widget.disabled = True
    #             widget.value = self.projections.metadata.metadata[name]

    #     if not all(widget.disabled for widget in self.widgets_to_enable):
    #         with self.metadata_input_output:
    #             display(self.metadata_widget_box)

    # def create_and_display_metadata_tables(self):
    #     # Implementation to create and display metadata tables based on current metadata
    #     pass

    # def reset_required_widgets(self):
    #     """
    #     Resets metadata widgets to default values. This also sets the metadata dict
    #     values (widget.value = val triggers metadata setting callbacks)
    #     """
    #     for name, val, widget in zip(
    #         self.required_parameters, self.init_required_values, self.widgets_to_enable
    #     ):
    #         if name not in self.projections.metadata.metadata:
    #             widget.value = val

    # def clear_output(self):
    #     self.metadata_input_output.clear_output()

    # def enter_metadata_output(self):
    #     """
    #     Enables/disables widgets if they are not/are already in the metadata. Displays
    #     the box if any of the widgets are not disabled.
    #     """

    #     # Zip params/initial values/widgets and set it to default if not in metadata
    #     self.required_metadata = zip(
    #         self.required_parameters, self.init_required_values, self.widgets_to_enable
    #     )
    #     # if required parameter not in current metadata instance, enable it and set
    #     # metadata to default value. if it is, disable and set value to the metadata
    #     # value
    #     for name, val, widget in self.required_metadata:
    #         if name not in self.projections.metadata.metadata:
    #             widget.disabled = False
    #             widget.value = val
    #             self.projections.metadata.metadata[name] = val
    #         else:
    #             widget.disabled = True
    #             widget.value = self.projections.metadata.metadata[name]

    #     # create metadata dataframe. The dataframe will only appear once all the
    #     # required metadata is inside the metadata instance
    #     self.create_and_display_metadata_tables()

    #     # pop up the widget box if any are disabled
    #     if not all([x.disabled for x in self.widgets_to_enable]):
    #         with self.metadata_input_output:
    #             display(self.metadata_widget_box)
