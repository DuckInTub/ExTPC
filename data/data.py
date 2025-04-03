from collections import defaultdict
import os
import scipy.io
import numpy as np
import json
import pickle
import pprint
import sys

with open("data.pkl", "rb") as file:
    data = pickle.load(file)
    pprint.pprint(data, depth=3)
sys.exit(1)

data_dict = defaultdict(lambda: defaultdict(dict))

for file in os.listdir("."):
    if not file.endswith(".mat"):
        continue
    person, speed = file.split("_")
    speed = speed[:-4]
    data = scipy.io.loadmat(file)
    data = data["data"]
    field_names = data.dtype.names

    for field in field_names:
        arr = data[field][0, 0].flatten().tolist()
        arr = [round(X, 2) for X in arr]
        data_dict[field][speed][person] = arr

print(data_dict.keys())


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        # Convert defaultdict to a regular dict
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        # If it's a list, convert each item
        d = [convert_defaultdict_to_dict(item) for item in d]
    return d

pickle_dict = convert_defaultdict_to_dict(data_dict)

with open("data.pkl", "wb") as file:
    pickle.dump(pickle_dict, file)