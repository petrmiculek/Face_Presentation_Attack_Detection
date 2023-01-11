# stdlib
import json
import time

# external
from prettytable import PrettyTable
import numpy as np

# local
# -

class LogTime:
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
    """
    Counts total number of trainable parameters of given torch model. Prints table of its layers.

    :param model: torch model of NN

    :return: Number of trainable parameters.
    """

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def get_dict(obj):
    return {key: value for key, value
            in obj.__dict__.items()
            if not key.startswith('_')}


def print_dict(dict_results):
    print('')  # newline
    for k, v in dict_results.items():
        print(f'\t{k:20s}:', end='')
        if isinstance(v, float):
            print(f' {v:06.4f}')
        elif isinstance(v, dict):
            print_dict(v)
        else:
            print(f' {v}')

def xor(a, b):
    return (a and not b) or (not a and b)

def keys_append(dictionary, suffix):
    return {k + suffix: v for k, v in dictionary.items()}


def save_config(union_dict, path):
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
