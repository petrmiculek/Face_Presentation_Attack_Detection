#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'

"""
Explanation methods.
- CAM
- perturbation metrics
- fake CAMs (Sobel, circle)
"""
# stdlib
# -

# external
import PIL
import cv2
import numpy as np

# local
from util_image import normalize
from config import blur_img_s as s_img, blur_mask_s as s_mask, blur_cam_s as s_cam, cam_blurred_weight


def perturbation_masks(cam, percentages):
    """ Get perturbation masks for deletion and insertion metrics.
    :param cam: CAM [h, w]
    :param percentages: list of percentages to keep
    :return: list of masks [h, w]
    mask values: 1 = keep, 0 = delete

    CAM is blurred to make the masks smoother. Since the blurred is only
    added to the original, the original values are not lost, and will
    remain as the highest values in the blurred CAM.
    """
    cam_smooth = cam + cam_blurred_weight * cv2.blur(cam, (s_cam, s_cam))
    thresholds = np.percentile(cam_smooth, percentages)

    if percentages[0] == 100:  # edge-case thresholds: 100% and 0%
        thresholds[0] = cam_smooth.max() + 1e-4  # ensure whole image kept
    if percentages[-1] == 0:
        thresholds[-1] = cam_smooth.min()  # ensure 0% leaves no pixels
    ''' Create masks for all perturbation levels. '''
    masks = []
    for th in thresholds:
        mask = np.float32(cam_smooth < th)
        # mask also blurred lightly to prevent sharp edges when applied
        mask_smooth = cv2.blur(mask, (s_mask, s_mask))
        masks.append(mask_smooth)
    return masks


def perturbation_baselines(img_np, which=None):
    """ Get perturbation baselines for deletion and insertion metrics.
    :param img_np: image [c, h, w]
    :param which: baselines to return, all by default
    :return: dict of numpy arrays float [0, 1], shape (C, H, W):
    default baselines (5): black-only, mean image color, and blurred
    input image with full, 1/4, and 1/8 of brightness, respectively.
    """
    if which is None:
        which = ['black', 'mean', 'blur', 'blur_div4', 'blur_div8']
    black = np.zeros_like(img_np)
    blurred = np.stack([cv2.blur(img_np[c], (s_img, s_img)) for c in range(3)])  # np.stack for CHW
    mean = np.zeros_like(img_np) + np.mean(img_np, axis=(1, 2), keepdims=True)
    div4 = blurred / 4
    div8 = blurred / 8
    baselines = {'black': black, 'blur': blurred, 'mean': mean,
                 'blur_div4': div4, 'blur_div8': div8}
    baselines = {k: baselines[k] for k in which}
    return baselines


def overlay_cam(img, cam, norm=True, interp=cv2.INTER_CUBIC):
    """ Overlay CAM on image.
    :param img: PIL jpeg (uint8) or numpy array (float [0,1]) (H, W, 3)
    :param cam: np array uint8, (H, W), e.g. (12,12)
    :param norm: rescale cam to [0, 1]
    :param interp: interpolation method (cv2.INTER_CUBIC, cv2.INTER_NEAREST)
    :return: np array uint8 (H, W, 3)

    - cubic interpolation looks the best, but it doesn't maintain the value range => clamp
    - viridis == default matplotlib colormap
    - how is the colormap applied? [0, 1] or [min, max]?
    - blending image_weight chosen empirically
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL.Image import Image as PILImage

    # normalize image and cam
    if isinstance(img, PILImage):
        img = np.array(img, dtype=float) / 255

    if norm:
        cam = normalize(cam)

    cam_min, cam_max = np.min(cam), np.max(cam)
    # resize cam to image size
    cam_np_resized = cv2.resize(cam, (img.shape[1], img.shape[0]),
                                interpolation=interp)
    # clamp to [min, max], as cubic doesn't keep the value range
    cam_np_resized = np.clip(cam_np_resized, cam_min, cam_max)
    overlayed = show_cam_on_image(img, cam_np_resized, use_rgb=True, image_weight=0.3,
                                  colormap=cv2.COLORMAP_VIRIDIS)
    return overlayed


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
    img = img.astype(np.float32)
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
    shapes = {'ResNet': (7, 7), 'EfficientNet': (12, 12)}
    return shapes[name]


class SobelFakeCAM:
    """
    Fake CAM - Sobel edge detector.

    CAM-like interface for a model-agnostic explanation.
    """

    def __init__(self, *args, **kwargs):
        self.shape = model_cam_shape(kwargs['model'])
        pass

    def __call__(self, input_tensor, *args, **kwargs):
        from torch import Tensor as torch_tensor
        if isinstance(input_tensor, torch_tensor):
            input_tensor = input_tensor[0].cpu().numpy().transpose(1, 2, 0)

        edges = sobel_edges(input_tensor)
        edges_lr = cv2.resize(edges, self.shape, interpolation=cv2.INTER_AREA)

        return edges_lr[None, ...]  # extra dim to match how CAMs are returned


class CircleFakeCAM:
    """
    Fake CAM - centered circle.
    CAM-like interface for a model-agnostic, data-agnostic explanation.
    """

    def __init__(self, *args, **kwargs):
        self.shape = model_cam_shape(kwargs['model'])
        self.cache = dict()

    def __call__(self, input_tensor, *args, **kwargs):
        # ignore input shape, use model-determined shape
        shape = self.shape
        if shape in self.cache:
            expl = self.cache[shape]
        else:
            expl = self.gaussian_circle(shape)
            self.cache[shape] = expl

        return expl[None, ...]

    def gaussian_circle(self, shape):
        center = (shape[0] // 2, shape[1] // 2)
        dists = np.sqrt((np.arange(0, shape[0])[:, None] - center[0]) ** 2 + (
                np.arange(0, shape[1])[None, :] - center[1]) ** 2)
        expl = normalize(-dists)
        return expl
