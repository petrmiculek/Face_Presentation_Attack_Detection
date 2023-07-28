#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'

"""
Utilities for images and plots
- normalize images
- general plotting functions
"""
import numpy as np
from PIL import Image

def deprocess(img):
    """ Normalize image for visualization. """
    from torch import Tensor as torch_tensor

    if isinstance(img, torch_tensor):
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)  # CHW -> HWC
    img -= np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img *= 0.1
    img += 0.5
    img = np.clip(img, 0, 1)
    # don't make image uint8
    return img


def normalize(img):
    """ Rescale image values to [0, 1]. """
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)


def mix_mask(img, mask, inside=1, outside=0.3):
    """ Highlight masked area in the image, darken the rest. """
    return img * outside + mask * img * (inside - outside)


def plot_many_df(df, *args, **kwargs):
    if 'path' not in df.columns:
        raise ValueError('plot_many_df: DataFrame must contain a column named "path"')
    if len(df) > 10:
        raise ValueError('plot_many_df: too many images (max 10)')
    imgs = [np.array(Image.open(path)) for path in df['path']]
    plot_many(imgs, *args, **kwargs)

def plot_many(*imgs, title=None, titles=None, output_path=None, show=True, rows=None, **kwargs):
    """
    Plot multiple images in a row/grid.

    :param imgs: list of images to plot
    :param title: figure title
    :param titles: per-image titles
    :param output_path: save figure to this path if not None
    :param show: toggle showing the figure
    :param rows: number of rows in the grid (if None, automatically determined)
    :param kwargs: keyword arguments for imshow
    """
    from torch import Tensor as torch_tensor
    from matplotlib import pyplot as plt
    from PIL.Image import Image as PILImage

    if len(imgs) == 1 and isinstance(imgs[0], (list, tuple)):
        # unwrap imgs object if necessary (should be passed as plot_many(*imgs),
        # but sometimes I forget and pass plot_many(imgs))
        imgs = imgs[0]
    imgs = list(imgs)  # if tuple, convert to list
    total = len(imgs)
    if rows is None:  # determine number of rows automatically
        rows = 1 if total < 4 else int(np.ceil(np.sqrt(total)))
        cols = int(np.ceil(total / rows))
        rows, cols = min(rows, cols), max(rows, cols)
    else:
        cols = int(np.ceil(total / rows))

    # fill rectangle with white 2x2 images if necessary
    if total < rows * cols:
        imgs.extend([np.ones((2, 2, 3))] * (rows * cols - total))

    fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    fig.suptitle(title)
    for i, img in enumerate(imgs):
        # select current axis
        if total == 1:
            ax_i = ax
        elif rows == 1:
            ax_i = ax[i]
        else:
            ax_i = ax[i // cols, i % cols]  # indexing correct, read properly!

        if isinstance(img, torch_tensor):
            img = np.array(img.cpu())

        if isinstance(img, PILImage):
            img = np.array(img)
        if img.ndim == 4:
            img = img[0, ...]
        if img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)

        ax_i.imshow(img, **kwargs)
        ax_i.axis('off')
        if titles is not None and i < len(titles):
            ax_i.set_title(titles[i])

    if rows == 2:
        # make more vertical space in between
        plt.subplots_adjust(hspace=0.3)

    plt.tight_layout(pad=0.8)
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
    try:
        idx = int(idx)
    except (ValueError, TypeError):
        idx = randint(0, len(markers))

    return markers[idx % len(markers)]
