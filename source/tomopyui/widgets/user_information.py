#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ipywidgets import *


def create_user_info_box(generalmetadata):
    extend_description_style = {"description_width": "auto"}
    collection_date = DatePicker(
        description="Date that you collected your data:",
        style=extend_description_style,
        disabled=False,
        layout=Layout(width="99%", font=30),
    )

    analysis_date = DatePicker(
        description="Date you are doing this analysis:",
        style=extend_description_style,
        disabled=False,
        layout=Layout(width="99%"),
    )

    user_name = Text(
        value="",
        style=extend_description_style,
        placeholder="X-ray Microscopist",
        description="Your name:",
        disabled=False,
        layout=Layout(width="99%"),
    )

    user_institution = Text(
        value="",
        style=extend_description_style,
        placeholder="Stanford Synchrotron Radiation Lightsource",
        description="Your institution:",
        disabled=False,
        layout=Layout(width="99%"),
    )

    phone_number = Text(
        value="",
        style=extend_description_style,
        placeholder="555-555-5555",
        description="Your phone number (for texting you when finished with jobs):",
        disabled=False,
        layout=Layout(width="99%"),
    )

    phone_number = Text(
        value="",
        style=extend_description_style,
        placeholder="555-555-5555",
        description="Your phone number (for texting you when finished with jobs):",
        disabled=False,
        layout=Layout(width="99%"),
    )

    carrier = Dropdown(
        value="Verizon",
        options=["Verizon", "T-Mobile"],
        style=extend_description_style,
        placeholder="555-555-5555",
        description="Your carrier:",
        disabled=False,
        layout=Layout(width="99%"),
    )
    email = Text(
        value="",
        style=extend_description_style,
        placeholder="user@slac.stanford.edu",
        description="Your email:",
        disabled=False,
        layout=Layout(width="99%"),
    )

    def update_collection_date(change):
        generalmetadata["collection_date"] = change.new
        generalmetadata["collection_date"] = str(
            generalmetadata["collection_date"]
        ).replace("-", "")

    def update_analysis_date(change):
        generalmetadata["analysis_date"] = change.new
        generalmetadata["analysis_date"] = str(
            generalmetadata["analysis_date"]
        ).replace("-", "")

    def update_user_name(change):
        generalmetadata["user_name"] = change.new

    def update_user_institution(change):
        generalmetadata["user_institution"] = change.new

    def update_phone_number(change):
        generalmetadata["phone_number"] = change.new

    def update_email(change):
        generalmetadata["email"] = change.new

    def update_carrier(change):
        generalmetadata["carrier"] = change.new

    user_name.observe(update_analysis_date, names="value")
    user_institution.observe(update_analysis_date, names="value")
    collection_date.observe(update_collection_date, names="value")
    analysis_date.observe(update_analysis_date, names="value")
    phone_number.observe(update_phone_number, names="value")
    carrier.observe(update_carrier, names="value")
    email.observe(update_email, names="value")

    box_layout = Layout(border="3px solid green", width="50%", align_items="stretch")
    infobox = VBox(
        children=[
            user_name,
            user_institution,
            phone_number,
            carrier,
            email,
            collection_date,
            analysis_date,
        ],
        layout=box_layout,
    )

    return generalmetadata, infobox
