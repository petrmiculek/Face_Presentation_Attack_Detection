# stdlib
import json
import time

# external
from prettytable import PrettyTable
import numpy as np

# local
# -

class LogTime:
    """Log times during run, print at the end"""
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.messages = []

    def log(self, msg):
        self.times.append(time.perf_counter())
        self.messages.append(msg)

    def print(self):
        # flag: from_index, new_only
        print('=' * 16, '\n'
                        ' i| time| message')
        for i, (t, msg) in enumerate(zip(self.times, self.messages)):
            print(f'{i:02d}| {(t - self.times[0]):2.2f}| {msg}')


def count_parameters(model):
    """Count total number of trainable parameters of a torch model. Prints table of its layers."""

    table = PrettyTable(["Modules", "Parameters"])
    params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        params += param

    print(table)
    print(f"Params#: {params}")

    return params


def get_dict(obj):
    """Get object's attributes as a dictionary."""
    return {key: value for key, value
            in obj.__dict__.items()
            if not key.startswith('_')}


def print_dict(dictionary, title=''):
    """Print dictionary formatted."""
    print(title)  # newline
    for k, v in dictionary.items():
        print(f'\t{k:20s}:', end='')
        if isinstance(v, float):
            print(f' {v:06.4f}')
        elif isinstance(v, dict):
            print_dict(v)
        else:
            print(f' {v}')

def xor(a, b):
    """XOR of two boolean values."""
    return (a and not b) or (not a and b)

def keys_append(dictionary, suffix):
    """Appends suffix to all keys in dictionary."""
    return {k + suffix: v for k, v in dictionary.items()}


def save_dict_json(union_dict, path):
    # make dictionary serializable as JSON
    for k, v in union_dict.items():
        if type(v) == np.ndarray:
            union_dict[k] = v.tolist()
        elif type(v) == np.int64:
            union_dict[k] = int(v)
        elif type(v) == np.float64:
            union_dict[k] = float(v)

    # save config to json
    with open(path, 'w') as f:
        json.dump(union_dict, f, indent=4)


def get_var_name(var, locals_foreigners):
    """Get variable name as string."""
    for k, v in locals_foreigners.items():
        if v is var:
            return k

    return 'unknown_var'
