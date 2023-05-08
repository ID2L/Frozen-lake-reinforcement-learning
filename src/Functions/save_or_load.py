from typing import Callable
import os
import numpy as np
import json

# https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_or_load(path: str, function: Callable):
    if os.path.isfile(path):
        with open(path, 'r') as the_file:
            results = json.loads(the_file.read())
    else:
        results = function()
        with open(path, 'w') as the_file:
            # Serializing json
            # Globally, may be faster here
            # See https://stackoverflow.com/a/57087055
            json_object = json.dumps(results, indent=4, cls = NpEncoder)
            the_file.write(json_object)
    return results
