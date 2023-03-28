"""
    :filename model_util.py (originally EarlyStopping.py)

    :brief EarlyStopping class file.

    Early stopping is used to avoid overfitting of the model.
    As the PyTorch library does not contain built-in early stopping, this class is from following repository:
    https://github.com/Bjarten/early-stopping-pytorch

    Original author:
    Bjarte Mehus Sunde, 2018

    Original author's mail:
    BjarteSunde@outlook.com

    Licence:
    MIT License

    Copyright (c) 2018 Bjarte Mehus Sunde

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    :author Tibor Kubik
    :author Petr Miculek

    :email xkubik34@stud.fit.vutbr.cz
    :email xmicul08@stud.fit.vutbr.cz

    File was created as a part of project 'Image super-resolution for rendered volumetric data' for POVa/2021Z course
    at the Brno University of Technology.
"""

import torch
import numpy as np
# from torchvision.models import resnet18  # unused, architecture loaded locally to allow changes
from torchvision.models import ResNet18_Weights
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
# local
from src.resnet18 import resnet18


def load_model(model_name, num_classes, seed=None):
    # todo seed is not used
    if model_name == 'resnet18':
        # load model with pretrained weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights, weight_class=ResNet18_Weights)
        # replace last layer with n-ary classification head
        model.fc = torch.nn.Linear(512, num_classes, bias=True)
        preprocess = weights.transforms()
    elif model_name == 'efficientnet_v2_s':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights,
                                  weight_class=EfficientNet_V2_S_Weights)
        dropout = 0.2  # as per original model code
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=True),
            torch.nn.Linear(1280, num_classes),
        )

        preprocess = weights.transforms()
    else:
        raise ValueError(f'Unknown model name {model_name}')
    return model, preprocess


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=1e-4, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.epoch_best = -1
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping: {self.counter} / {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.4f} -> {val_loss:.4f}).')

        if self.path is not None:
            torch.save(model.state_dict(), self.path)
            self.trace_func('Saving model ...')

        self.val_loss_min = val_loss
