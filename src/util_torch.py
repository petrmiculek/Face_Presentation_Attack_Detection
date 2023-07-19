# stdlib
import os
from os.path import join

# external
import torch
import numpy as np
from prettytable import PrettyTable
from torch.nn import functional as F
from torch.nn import Dropout
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_V2_S_Weights  # unused: efficientnet_v2_s, subclassed instead

from tqdm import tqdm

from efficientnet_v2_s import efficientnet_v2_s
# local
from resnet18 import resnet18
from augmentation import ClassificationPresetTrain, ClassificationPresetEval


def load_model(model_name, num_classes, freeze_backbone=False):
    """ Load model, replace classification head, maybe freeze backbone. """

    if model_name == 'resnet18':
        # load model with pretrained weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights, weight_class=ResNet18_Weights)
        # replace last layer with n-ary classification head
        transforms_orig = weights.transforms()

    elif model_name == 'efficientnet_v2_s':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights,
                                  weight_class=EfficientNet_V2_S_Weights)
        transforms_orig = weights.transforms()
    else:
        raise ValueError(f'Unknown model name {model_name}')
    ''' Set model to binary or multiclass '''
    if num_classes == 2:
        model.switch_to_binary()
    else:
        model.switch_to_multiclass()

    ''' Disable in-place operations '''
    if True:
        # explanations work with forward hooks, inplace operations break them
        inplace_c = 0
        inplace_n = []
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU) or \
                    isinstance(module, torch.nn.SiLU) or \
                    isinstance(module, Dropout):
                module.inplace = False
                inplace_c += 1
                inplace_n.append(module.__class__.__name__)

        if inplace_c > 0:
            print(f'Disabled inplace operations for {inplace_c} modules')

    ''' Freeze backbone parameters '''
    if freeze_backbone:
        layer_to_unfreeze = ''
        if model_name == 'resnet18':
            layer_to_unfreeze = 'fc'
        elif model_name == 'efficientnet_v2_s':
            layer_to_unfreeze = 'classifier'  # or features.7

        # print('Note: Currently not freezing any layers')
        for name, param in model.named_parameters():
            if layer_to_unfreeze not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            print(name, param.requires_grad)

    ''' Preprocessing and Augmentations '''
    crop_size = transforms_orig.crop_size[0]
    resize_size = transforms_orig.resize_size[0]
    transform_train = ClassificationPresetTrain(auto_augment_policy='ta_wide', random_erase_prob=0.5,
                                                crop_size=crop_size)
    transform_eval = ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size)
    preprocess = {
        'crop_size': crop_size,
        'resize_size': resize_size,
        'train': transform_train,
        'eval': transform_eval,
    }
    return model, preprocess


def load_model_eval(model_name, num_classes, run_dir, device='cuda:0'):
    """ Load Model """
    import config

    model_name = model_name
    model, preprocess = load_model(model_name, num_classes)
    model.load_state_dict(torch.load(join(run_dir, 'model_checkpoint.pt'), map_location=device), strict=False)
    model.to(device)
    model.eval()
    # sample model prediction
    out = model(torch.rand(config.sample_shape).to(device)).shape
    # assert shape is (1, num_classes)
    assert out == (1, num_classes), f'Model output shape is {out}'

    return model, preprocess


def init_device():
    """ Initialize device, print setup, set printing format precision. """
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    # Human-readable printing for Numpy + Torch
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    return device


def init_seed(seed=None):
    """ Set random seed for reproducibility. """
    if seed is None:
        print('No random seed set')
    else:
        print(f'Random seed: {seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_module(dataset_name):
    """
    Import dataset module.
    Name-based match for the .py module, specific split is defined later.
    Placing the code here is a workaround to avoid circular imports.
    """
    dataset_name = dataset_name.split('-')[0]  # ignore suffix after '-' (e.g. 'rose_youtu-full')
    if dataset_name == 'rose_youtu':
        import dataset_rose_youtu as dataset_module
    elif dataset_name == 'siwm':
        import dataset_siwm as dataset_module
    else:
        raise ValueError(f'Unknown dataset name {dataset_name}')
    return dataset_module


def count_parameters(model, sum_only=False):
    """
    Count total number of trainable parameters of a torch model. Prints table of its layers.

    Taken from a previous own project, original source unknown.
    """

    table = PrettyTable(["Modules", "Parameters"])
    params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        params += param

    if not sum_only:
        print(table)
    # print number of params in exponential notation
    print(f"Params#: {params:.3e}")

    return params


def predict(model, inputs):
    """ Predict on batch, return Numpy softmax probabilities and classes. """
    with torch.no_grad():
        preds_raw = model(inputs)
        probs = F.softmax(preds_raw, dim=1).cpu().numpy()
        classes = np.argmax(probs, axis=1)
    return probs, classes


def eval_loop(model, loader, criterion, device, desc='Eval'):
    """ Evaluate model on dataset. """
    len_loader = len(loader)
    ep_loss = 0.0
    preds, labels, paths = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, mininterval=1., desc=desc)
        for sample in pbar:
            img, label = sample['image'], sample['label']
            labels.append(label)
            paths.append(sample['path'])
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            out = model(img)
            loss = criterion(out, label)
            ep_loss += loss.item()
            prediction_hard = torch.argmax(out, dim=1).cpu().numpy()
            preds.append(prediction_hard)
            pbar.set_postfix(loss=f'{loss:.4f}', refresh=False)

    ep_loss /= len_loader  # loss is averaged over batch already, divide by number of batches
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    return paths, labels, preds, ep_loss


def change_relu_resnet(model):
    """
    Change inplace ReLU to non-inplace ReLU in ResNet model.
    source: https://github.com/LenkaTetkova/robustness-of-explanations/blob/433fca431c13a1cc1e1c059fbac05685637ee391/src/models/torch_model.py#L109
    """

    model.relu = torch.nn.ReLU(inplace=False)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for key in layer._modules.keys():
            layer._modules[key].relu = torch.nn.ReLU(inplace=False)
    return model


def change_silu_efficientnet(model):
    """
    Change inplace SiLU to non-inplace SiLU in EfficientNet model.
    source: https://github.com/LenkaTetkova/robustness-of-explanations/blob/433fca431c13a1cc1e1c059fbac05685637ee391/src/models/torch_model.py#L109
    """
    for i in range(len(model.features._modules)):
        for j in range(len(model.features._modules[str(i)]._modules)):
            module_j = model.features._modules[str(i)]._modules[str(j)]
            if isinstance(module_j, torch.nn.SiLU):
                module_j = torch.nn.SiLU(inplace=False)
            elif 'block' in module_j._modules.keys():
                k_max = len(module_j._modules['block']._modules)
                for k in range(k_max):
                    module_k = module_j._modules['block']._modules[str(k)]._modules
                    m_max = len(module_k)
                    for m in range(m_max):
                        name_matches_1 = str(m) in module_k.keys()
                        type_matches_1 = isinstance(module_k[str(m)], torch.nn.SiLU)
                        name_matches_2 = "activation" in module_k.keys()
                        type_matches_2 = isinstance(module_k['activation'], torch.nn.SiLU)

                        if (name_matches_1 and type_matches_1):
                            module_k[str(m)] = torch.nn.SiLU(inplace=False)
                        elif (name_matches_2 and type_matches_2):
                            module_k['activation'] = torch.nn.SiLU(inplace=False)
    model.classifier._modules['0'] = Dropout(p=0.3, inplace=False)
    torch.autograd.set_grad_enabled(True)
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features.parameters():
        param.requires_grad = True
    return model


"""
    :filename util_torch.py (originally EarlyStopping.py)

    Early Stopping adapted from: vvvvvvv

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
"""
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
