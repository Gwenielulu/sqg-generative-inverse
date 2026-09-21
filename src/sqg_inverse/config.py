from copy import deepcopy
from pathlib import Path
import yaml


def read(path):
    with Path(path).expanduser().resolve().open() as f:
        return yaml.safe_load(f)


def merge(a, b):
    out = deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def root():
    return Path(__file__).resolve().parents[2]
