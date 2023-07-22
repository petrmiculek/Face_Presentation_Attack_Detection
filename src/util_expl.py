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
    cam_blurred = cam + cam_blurred_weight * cv2.blur(cam, (s_cam, s_cam))
    thresholds = np.percentile(cam_blurred, percentages)
    thresholds[-1] = cam_blurred.min()  # make sure 0% kept is 0
    ''' Levels of perturbation '''
    masks = []
    for th in thresholds:
        mask = np.float32(cam_blurred < th)
        mask_blurred = cv2.blur(mask, (s_mask, s_mask))
        masks.append(mask_blurred)
    return masks


def perturbation_baselines(cam, img_np, which=None):
    """ Get perturbation baselines for deletion metric.
    :param cam: CAM [h, w]
    :param img_np: image [c, h, w]
    :param which: baselines to return
    :return: dict with deletion metric baselines
    baselines are numpy arrays float [0, 1], shape (C, H, W): C=1 for black, C=3 for blur and mean
    """

    if which is None:
        which = ['black', 'blur', 'mean', 'blurdark', 'blurdarker', 'blurdarkerer']
    b_black = np.zeros_like(cam)[None, ...]
    b_blur = np.stack([cv2.blur(img_np[c], (s_img, s_img)) for c in range(3)])  # np.stack for CHW
    b_mean = np.zeros_like(img_np) + np.mean(img_np, axis=(1, 2), keepdims=True)
    b_blurdark = b_blur / 4
    b_blurdarker = b_blur / 6
    b_blurdarkerer = b_blur / 8
    baselines_all = {'black': b_black, 'blur': b_blur, 'mean': b_mean, 'blurdark': b_blurdark,
                     'blurdarker': b_blurdarker, 'blurdarkerer': b_blurdarkerer}
    baselines = {k: baselines_all[k] for k in which}
    return baselines


def overlay_cam(img, cam, norm=True, interp=cv2.INTER_CUBIC):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL.Image import Image as PILImage
    """
    Overlay CAM on image.
    :param img: PIL jpeg (uint8) or numpy array (float [0,1]) (H, W, 3)
    :param cam: np array uint8, (H, W), e.g. (12,12)
    :param norm: rescale cam to [0, 1]
    :param interp: interpolation method (cv2.INTER_CUBIC, cv2.INTER_NEAREST)
    :return: np array uint8 (H, W, 3)

    - cubic looks better but it doesn't maintain the value range => clamp or rescale
    - viridis = default matplotlib colormap
    - how is the colormap applied? [0, 1] or [min, max]?
    - blending image_weight chosen empirically
    """

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

        return expl[None, ...]

    def gaussian_circle(self, shape):
        center = (shape[0] // 2, shape[1] // 2)
        dists = np.sqrt((np.arange(0, shape[0])[:, None] - center[0]) ** 2 + (
                np.arange(0, shape[1])[None, :] - center[1]) ** 2)
        expl = normalize(-dists)
        return expl
