import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import Tensor as torch_tensor


def overlay_cam(img, cam):
    """
    Overlay CAM on image.
    :param img: PIL jpeg ~uint8?, WxHx3
    :param cam: np array uint8, MxN (e.g. 12x12)
    :return: np array uint8, WxHx3

    - cubic looks better but it doesn't maintain the value range => clamp or rescale
    - viridis = default matplotlib colormap
    - todo: how is the colormap applied? [0, 1] or [min, max]?
    - blending weight arbitrary
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
    if isinstance(img, torch_tensor):
        img = img.detach().cpu().numpy()  # might fail when no grad?
        img = img.transpose(1, 2, 0)
    img -= np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img *= 0.1
    img += 0.5
    img = np.clip(img, 0, 1)
    # don't make image uint8
    return img
