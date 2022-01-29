#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ipywidgets import *

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


def import_module_set_env(import_dict):
    """
    From https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed

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
