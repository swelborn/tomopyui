import os
import glob
import numpy as np
import json
import functools
import tifffile as tf
import asyncio
import logging
import ipywidgets as widgets
import importlib.util
import sys
import time

from ipywidgets import *
from abc import ABC, abstractmethod


def import_module_set_env(import_dict):
    """
    https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed

    Safely imports a module or package and sets an environment variable if it
    imports (or is already imported). This is used in the main function for
    checking whether or not `cupy` is installed. If it is not installed, then
    options for cuda-enabled functions will be greyed out.
    """
    for key in import_dict:
        if key in sys.modules:
            os.environ[import_dict[key]] = "True"
            pass
        elif (spec := importlib.util.find_spec(key)) is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
            os.environ[import_dict[key]] = "True"
        else:
            os.environ[import_dict[key]] = "False"
            pass


# From ipywidgets readthedocs
class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {"width": "100%", "height": "160px", "border": "1px solid black"}
        self.out = Output(layout=layout)

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """Show the logs"""
        display(self.out)

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()


def return_handler(logger, logging_level=None):
    handler = OutputWidgetHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  - [%(levelname)s] %(message)s")
    )
    # handler.show_logs()
    logger.addHandler(handler)
    logger.setLevel(logging_level)  # log at info level.
    return handler, logger


class MetaCheckbox:
    def __init__(self, description, dictionary, obj, disabled=False, value=False):

        self.checkbox = Checkbox(
            description=description, value=value, disabled=disabled
        )

        def create_opt_dict_on_check(change):
            dictionary[description] = change.new
            obj.set_metadata()  # obj needs a set_metadata function

        self.checkbox.observe(create_opt_dict_on_check, names="value")


def create_checkbox(description, disabled=False, value=False):
    checkbox = Checkbox(description=description, disabled=disabled, value=value)
    return checkbox


def create_checkboxes_from_opt_list(opt_list, dictionary, obj):
    checkboxes = [MetaCheckbox(opt, dictionary, obj) for opt in opt_list]
    return [a.checkbox for a in checkboxes]  # return list of checkboxes


def set_checkbox_bool(checkbox_list, dictionary, obj):
    def create_opt_dict_on_check(change):
        dictionary[change.owner.description] = change.new
        obj.set_metadata()  # obj needs a set_metadata function

    for key in dictionary:
        if dictionary[key]:
            for checkbox in checkbox_list:
                if checkbox.description == str(key):
                    checkbox.value = True
                    checkbox.observe(create_opt_dict_on_check, names="value")
        elif not dictionary[key]:
            for checkbox in checkbox_list:
                if checkbox.description == str(key):
                    checkbox.value = False
                    checkbox.observe(create_opt_dict_on_check, names="value")
    return checkbox_list


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """Decorator that will postpone a function's
    execution until after `wait` seconds
    have elapsed since the last time it was invoked."""

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()

        return debounced

    return decorator


class ReactiveButtonBase(ABC):
    """
    Base class for a reactive button.

    Parameters
    ----------
    description: str
        Button description, initial.
    description_during: str
        Button description, during reaction.
    description_after: str
        Button description, during reaction.
    icon: str
        FontAwesome icon, initial.
    icon_during: str
        FontAwesome icon, during reaction.
    icon_after: str
        FontAwesome icon, after reaction.
    button_style: str
        Changes button color, initial. Can be "info", "success", "", "warning",
        or "danger"
    button_style_during: str
        Changes button color, during callback. Can be "info", "success", "", "warning",
        or "danger"
    button_style_after: str
        Changes button color, after callback. Can be "info", "success", "", "warning",
        or "danger"
    """

    def __init__(
        self,
        callback,
        description="",
        description_during="",
        description_after="",
        icon="",
        icon_during="fas fa-cog fa-spin fa-lg",
        icon_after="fa-check-square",
        button_style="",
        button_style_during="info",
        button_style_after="success",
        style=None,
        disabled=False,
        layout=None,
        tooltip=None,
        warning="That button click didn't work.",
    ):
        self.button = Button()
        self.callback = callback
        self.description = description
        self.description_during = description_during
        self.description_after = description_after
        self.button_style = button_style
        self.button_style_during = button_style_during
        self.button_style_after = button_style_after
        self.icon = icon
        self.icon_during = icon_during
        self.icon_after = icon_after
        self.disabled = disabled
        self.button_style_warning = "warning"
        self.layout = layout
        self.style = style
        self.tooltip = tooltip
        self.warning = warning
        self.reset_state()
        self.button.on_click(self.run_callback)

    def reset_state(self):
        self.button.description = self.description
        self.button.button_style = self.button_style
        self.button.icon = self.icon
        self.button.disabled = self.disabled
        self.button.layout = self.layout
        self.button.tooltip = self.tooltip
        if self.style is not None:
            self.button.style = self.style
        if self.layout is not None:
            self.button.layout = self.layout

    def switch_disabled(self):
        if self.disabled:
            self.disabled = False
            self.button.disabled = False
        else:
            self.disabled = True
            self.button.disabled = True

    def disable(self):
        self.reset_state()
        self.button.disabled = True

    def enable(self):
        self.reset_state()
        self.button.disabled = False

    def warning(self, *args):
        self.button.button_style = self.button_style_warning
        self.button.description = self.warning
        self.button.icon = "exclamation-triangle"

    @abstractmethod
    def run_callback(self, *args):
        ...


class ReactiveTextButton(ReactiveButtonBase):
    def __init__(
        self,
        callback,
        description,
        description_during,
        description_after,
        warning="That button didn't work.",
        layout=Layout(width="auto", height="auto", align_items="stretch"),
    ):
        super().__init__(
            callback,
            description=description,
            description_during=description_during,
            description_after=description_after,
            layout=layout,
            warning=warning,
        )

    def run_callback(self, *args):
        self.button.button_style = self.button_style_during
        self.button.icon = self.icon_during
        self.button.description = self.description_during
        self.callback()
        self.button.button_style = self.button_style_after
        self.button.icon = self.icon_after
        self.button.description = self.description_after


class ReactiveIconButton(ReactiveButtonBase):
    def __init__(self, callback, icon, icon_during, icon_after, skip_during):
        super().__init__(
            callback, icon=icon, icon_during=icon_during, icon_after=icon_after
        )

    def run_callback(self, *args):
        self.button.button_style = self.button_style_during
        self.button.icon = self.icon_during
        self.callback()
        self.button.button_style = self.button_style_after
        self.button.icon = self.icon_after


class SwitchOffOnIconButton(ReactiveButtonBase):
    """
    Subclass for buttons that turn off and on (green on, grey off).
    """

    def __init__(self, callback_on, callback_off, icon):
        super().__init__(None, icon=icon, icon_during=icon, icon_after=icon)
        self.callback_on = callback_on
        self.callback_off = callback_off
        self.button_on = False

    def run_callback(self, *args):
        if self.button_on:
            self.callback_off()
            self.button.button_style = ""
            self.button_on = False
        else:
            self.callback_on()
            self.button.button_style = "success"
            self.button_on = True


class ImportButton(ReactiveButtonBase):
    """
    Import button found throughout the app.
    """

    def __init__(self, callback):
        super().__init__(
            callback,
            icon="upload",
            tooltip="Load your data into memory",
            style={"font_size": "35px"},
            layout=Layout(width="75px", height="86px"),
            disabled=True,
        )
        self.button_on = False

    def run_callback(self, *args):
        self.button.button_style = self.button_style_during
        self.button.icon = self.icon_during
        self.button.description = self.description_during
        self.callback()
        self.button.button_style = self.button_style_after
        self.button.icon = self.icon_after
        self.button.description = self.description_after

    def switch_disabled(self):
        super().switch_disabled()
        if self.disabled:
            self.button.button_style = "info"
        else:
            self.button.button_style = ""

    def enable(self):
        self.disabled = False
        self.button.button_style = self.button_style
        self.button.disabled = False
        self.button.icon = self.icon
        self.button.button_style = self.button_style_during
