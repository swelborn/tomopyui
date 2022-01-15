# https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
import json
import os
import pandas as pd


def safe_serialize(obj, f):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, f, default=default, indent=4)


def save_metadata(filename, metadata):
    with open(filename, "w+") as f:
        a = safe_serialize(metadata, f)


def load_metadata(filepath=None, filename=None, fullpath=None):
    if fullpath is not None:
        with open(fullpath) as f:
            metadata = json.load(f)
    else:
        fullpath = os.path.abspath(os.path.join(filepath, filename))
        with open(fullpath) as f:
            metadata = json.load(f)
    return metadata


def metadata_to_DataFrame(metadata):
    metadata_frame = {}
    time, title = parse_printed_time(metadata["analysis_time"])
    extra_headers = ["Prj X Range", "Prj Y Range", "Start Angle", "End Angle", title]
    metadata_frame["Headers"] = list(metadata["opts"].keys())
    metadata_frame["Headers"] = [
        metadata_frame["Headers"][i].replace("_", " ").title().replace("Num", "No.")
        for i in range(len(metadata_frame["Headers"]))
    ]
    metadata_frame["Headers"] = metadata_frame["Headers"] + extra_headers
    extra_values = [
        metadata["prj_range_x"],
        metadata["prj_range_y"],
        metadata["angle_start"],
        metadata["angle_end"],
        time,
    ]
    extra_values = [str(extra_values[i]) for i in range(len(extra_values))]
    metadata_frame["Values"] = [
        str(metadata["opts"][key]) for key in metadata["opts"]
    ] + extra_values
    metadata_frame = {
        metadata_frame["Headers"][i]: metadata_frame["Values"][i]
        for i in range(len(metadata_frame["Headers"]))
    }
    sr = pd.Series(metadata_frame)
    df = pd.DataFrame(sr).transpose()
    s = df.style.hide_index()
    s.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.2em;"},
            {"selector": "td", "props": "text-align: center;" "font-size: 1.2em; "},
        ],
        overwrite=False,
    )
    return s


def parse_printed_time(timedict):
    if timedict["hours"] < 1:
        if timedict["minutes"] < 1:
            time = timedict["seconds"]
            title = "Time (s)"
        else:
            time = timedict["minutes"]
            title = "Time (min)"
    else:
        time = timedict["hours"]
        title = "Time (h)"

    time = f"{time:.1f}"
    return time, title
