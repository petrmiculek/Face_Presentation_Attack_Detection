from functools import partial
from typing import Sequence, Union, Optional, Any

import torch
from torch import nn
from torch.nn import Sequential, Dropout, Linear
from torchvision.models import EfficientNet as EfficientNetBase, EfficientNet_V2_S_Weights
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig, _efficientnet_conf


class EfficientNet(EfficientNetBase):
    # adapted from torchvision implementation, to subclass and add forward_train
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dropout = 0.2  # as per original model code
        ''' Create custom classifier heads '''
        self.classifier_binary = Sequential(
            Dropout(p=dropout, inplace=False),
            Linear(1280, 2), )

        self.classifier_multiclass = Sequential(
            Dropout(p=dropout, inplace=False),
            Linear(1280, 5), )

        self.classifier_rose = Sequential(
            Dropout(p=dropout, inplace=False),
            Linear(1280, 8), )

    def forward_train(self, x):
        """ Forward pass for training. """
        # rewrite base forward, add binary + n-ary head
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # embedding
        y_binary = self.classifier_binary(x)  # prediction (multiclass, 2 classes)
        y_multiclass = self.classifier_multiclass(x)  # prediction (multiclass, 5 classes)
        y_rose = self.classifier_rose(x)  # prediction (multiclass, 8 classes)
        return {'bin': y_binary, 'unif': y_multiclass, 'orig': y_rose}

    def forward_binary(self, x):
        """ Forward pass for binary classification (2 output classes, not sigmoid). """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_binary = self.classifier_binary(x)
        return y_binary

    def forward_multiclass(self, x):
        """ Forward pass for multi-class classification (5 output classes). """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_multiclass = self.classifier_multiclass(x)
        return y_multiclass

    forward = forward_multiclass  # default forward function

    def forward_rose(self, x):
        """ Forward pass for multi-class classification into RoseYoutu classes (8). """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_rose = self.classifier_rose(x)
        return y_rose

    def switch_to_binary(self):
        self.forward = self.forward_binary
        print('Model default predict switched to binary')

    def switch_to_multiclass(self):
        self.forward = self.forward_multiclass
        print('Model default predict switched to multiclass')

    def switch_to_rose(self):
        self.forward = self.forward_rose
        print('Model default predict switched to rose')


def _efficientnet(
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        last_channel: Optional[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    return model


def efficientnet_v2_s(
        *, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    """
    weights = EfficientNet_V2_S_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )
