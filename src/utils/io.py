import os
import glob
import uuid
from datetime import datetime
import yaml

def gather_paths(root, classes, split):
    paths, ys = [], []
    for ci, c in enumerate(classes):
        ps = sorted(glob.glob(os.path.join(root, split, c, '*')))
        paths.extend(ps)
        ys.extend([ci] * len(ps))
    return paths, __list_to_numpy(ys)

def __list_to_numpy(lst):
    import numpy as np
    return np.array(lst, dtype=int)

def make_run_id():
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"

def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_v{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def save_yaml(d, path):
    with open(path, 'w') as f:
        yaml.safe_dump(d, f)
