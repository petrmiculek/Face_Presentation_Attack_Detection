# stdlib
import json
import os
import time
from collections import defaultdict

# external
import numpy as np

import config


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
        np.save(path, file)


def plot_many(*imgs, title=None, titles=None, output_path=None, show=True, **kwargs):
    """
    Plot multiple images in a grid.

    :param imgs: list of images to plot
    :param title: figure title
    :param titles: per-image titles
    :param output_path: save figure to this path if not None
    :param show: toggle showing the figure
    :param kwargs: keyword arguments for imshow
    """
    import torch
    from matplotlib import pyplot as plt
    from PIL.Image import Image

    if len(imgs) == 1 and isinstance(imgs[0], (list, tuple)):
        # unwrap imgs object if necessary (should be passed as plot_many(*imgs),
        # but sometimes I forget and pass plot_many(imgs))
        imgs = imgs[0]
    imgs = list(imgs)  # if tuple, convert to list
    total = len(imgs)
    rows = 1 if total < 4 else int(np.ceil(np.sqrt(total)))
    cols = int(np.ceil(total / rows))
    rows, cols = min(rows, cols), max(rows, cols)

    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if title is not None:
        fig.suptitle(title)  # y=0.77 # height modification unused

    # fill imgs with white 2x2 images if necessary
    if total < rows * cols:
        imgs.extend([np.ones((2, 2, 3))] * (rows * cols - total))

    # for loop over axes
    for i, img in enumerate(imgs):
        # select current axis
        if total == 1:
            ax_i = ax
        elif rows == 1:
            ax_i = ax[i]
        else:
            ax_i = ax[i // cols, i % cols]  # indexing correct, read properly!

        if isinstance(img, torch.Tensor):
            img = np.array(img.cpu())

        if isinstance(img, Image):
            img = np.array(img)
        if img.ndim == 4:
            img = img[0, ...]
        if img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)

        ax_i.imshow(img, **kwargs)
        ax_i.axis('off')
        if titles is not None and i < len(titles):
            ax_i.set_title(titles[i])

    plt.tight_layout(pad=0.5)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    if show:
        plt.show()
    plt.close()


def get_marker(idx=None):
    """Get a marker for matplotlib plot."""
    from random import randint
    markers = ['o', 's', 'v', '^', 'D', 'P', 'X', 'h', 'd', 'p', 'H', '8', '>', '<', '*', 'x', 'o', 's', 'v', '^', 'D',
               'P', 'X', 'h', 'd', 'p', 'H', '8', '>', '<', '*', 'x']
    if idx is None:
        idx = randint(0, len(markers))

    return markers[idx % len(markers)]


def update_config(args_dict, global_vars=True, hparams=True):
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
