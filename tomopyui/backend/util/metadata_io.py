# https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
import json
import os


def safe_serialize(obj, f):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, f, default=default, indent=4)


def save_metadata(filename, metadata):
    with open(filename, "w+") as f:
        a = safe_serialize(metadata, f)

def load_metadata(filepath, filename):
    fp = os.path.abspath(os.path.join(filepath, filename))
    with open(fp) as f:
        metadata = json.load(f)
    return metadata