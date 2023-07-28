#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'

"""
Utility functions
- logging
- transforming data
- saving/loading
"""
# stdlib
import json
import os
import time
from collections import defaultdict

# external
import numpy as np
import pandas as pd



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
        print('=' * 16, '\n',
              ' i| time| message')
        for i, (t, msg) in enumerate(zip(self.times, self.messages)):
            print(f'{i:02d}| {(t - self.times[0]):2.2f}| {msg}')


def get_dict(obj):
    """Get object's attributes as a dictionary."""
    return {key: value for key, value
            in obj.__dict__.items()
            if not key.startswith('_')}


def dol_from_lod(lod):
    """Get dict of lists from list of dicts."""
    dol = defaultdict(list)
    for d in lod:
        for k, v in d.items():
            dol[k].append(v)
    return dol


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
    if path is None:
        print('Not saving config to json, path is None.')
        return

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


def save_i(path, file, overwrite=False):
    """ Save but don't overwrite """
    exists = os.path.exists(path)
    if exists and not overwrite:
        print(f'File {path} exists, skipping saving.')
    else:
        if exists:
            print(f'File {path} exists, overwriting.')
        if isinstance(file, np.ndarray):
            np.save(path, file)
        elif isinstance(file, pd.DataFrame):
            file.to_pickle(path)
        else:
            raise ValueError(f'Unknown type {type(file)}')

def update_config(args_dict, global_vars=True, hparams=True):
    """ Copy args to config (project-specific). """
    import config
    # update config with args
    if global_vars:
        for k, v in vars(config).items():
            if k in args_dict:
                setattr(config, k, args_dict[k])
                # print(f'Updated config.{k} = {v} -> {args_dict[k]}')
            elif k in config.HPARAMS and not hparams:
                # copy hparams to config
                setattr(config, k, config.HPARAMS[k])

    # update config.HPARAMS with args
    if hparams:
        for k, v in args_dict.items():
            if k in config.HPARAMS:
                config.HPARAMS[k] = v
                # print(f'Updated config.HPARAMS.{k} = {v}')
