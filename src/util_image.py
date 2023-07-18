import cv2
import numpy as np


def overlay_cam(img, cam):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    """
    Overlay CAM on image.
    :param img: PIL jpeg ~uint8?, WxHx3
    :param cam: np array uint8, MxN (e.g. 12x12)
    :return: np array uint8, WxHx3

    - cubic looks better but it doesn't maintain the value range => clamp or rescale
    - viridis = default matplotlib colormap
    - how is the colormap applied? [0, 1] or [min, max]?
    - blending image_weight chosen empirically
    """

    # normalize image and cam
    img_np = np.array(img, dtype=float) / 255
    cam_np = np.array(cam, dtype=float) / 255
    cam_min, cam_max = cam_np.min(), cam_np.max()
    # resize cam to image size
    cam_np_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]),
                                interpolation=cv2.INTER_CUBIC)  # INTER_NEAREST
    # clamp to [min, max], as cubic doesn't keep the value range
    cam_np_resized = np.clip(cam_np_resized, cam_min, cam_max)
    overlayed = show_cam_on_image(img_np, cam_np_resized, use_rgb=True, image_weight=0.3,
                                  colormap=cv2.COLORMAP_VIRIDIS)
    return overlayed


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


def sobel_edges(img):
    """
    Detect image edges (Sobel filter).

    Adapted from: https://docs.opencv.org/4.8.0/d2/d2c/tutorial_sobel_derivatives.html

    parameters `k` etc. chosen empirically based on few manually observed images,
    aiming for a rough outline of the contours.

    :param img: np array float32, WxHx3
    :return: np array float32, WxHx3
    """
    # grayscale, blur, sobel (x + y), sum, normalize
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    img_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    img_x = cv2.convertScaleAbs(img_x)
    img_y = cv2.convertScaleAbs(img_y)
    img = normalize(img_x + img_y)
    return img


def model_cam_shape(model):
    name = model.__class__.__name__
    shapes = {'resnet18': (7, 7), 'efficientnet_v2_s': (12, 12)}
    return shapes[name]


class SobelFakeCAM:
    """
    Fake CAM - Sobel edge detector.

    CAM-like interface for a model-agnostic, training-data-agnostic explanation.
    """

    def __init__(self, *args, **kwargs):
        self.shape = model_cam_shape(kwargs['model'])
        pass

    def __call__(self, input_tensor, *args, **kwargs):
        import torch
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor[0].cpu().numpy().transpose(1, 2, 0)

        edges = sobel_edges(input_tensor)
        edges_lr = cv2.resize(edges, self.shape, interpolation=cv2.INTER_AREA)

        return edges_lr


class CircleFakeCAM:
    """
    Fake CAM - centered circle.

    CAM-like interface for a model-agnostic, (all-) data-agnostic explanation.
    """

    def __init__(self, *args, **kwargs):
        self.shape = model_cam_shape(kwargs['model'])
        self.cache = dict()

    def __call__(self, input_tensor, *args, **kwargs):
        # shape = input_tensor.shape
        # if input_tensor.ndim == 4:
        #     shape = shape[0]
        # shape = (1, shape[2], shape[3])
        # ignore input shape, use model-determined shape
        shape = self.shape
        if shape in self.cache:
            expl = self.cache[shape]
        else:
            expl = self.gaussian_circle(shape)
            self.cache[shape] = expl

        return expl

    def gaussian_circle(self, shape):
        center = (shape[0] // 2, shape[1] // 2)
        dists = np.sqrt((np.arange(0, shape[0])[:, None] - center[0]) ** 2 + (
                np.arange(0, shape[1])[None, :] - center[1]) ** 2)
        expl = normalize(-dists)
        return expl


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
    from torch import Tensor as torch_tensor
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

        if isinstance(img, torch_tensor):
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
