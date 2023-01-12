# stdlib
import argparse
import json
import logging
import os
import datetime
from os.path import join
from copy import deepcopy

"""
# fix for local import problems - add all local directories
import sys
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)
"""

# external
import matplotlib.pyplot as plt
import torch
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import ResNet18_Weights
import numpy as np
from tqdm import tqdm
import pandas as pd

os.environ["WANDB_SILENT"] = "true"

import wandb as wb

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
import dataset_rose_youtu as dataset
from metrics import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, xor, keys_append, save_config
from model_util import EarlyStopping
import resnet18

run_dir = 'runs/2023-01-10_14-41-03'

model = None
device = None
preprocess = None
criterion = None


def eval_loop(loader):
    len_loader = len(loader)
    ep_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for img, label in tqdm(loader, leave=False, mininterval=1.):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, label)
            ep_loss += loss.item()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)

            # save prediction
            labels.append(label.cpu().numpy())
            preds.append(prediction_hard.cpu().numpy())

    # loss is averaged over batch already, divide by batch number
    ep_loss /= len_loader

    metrics = compute_metrics(labels, preds)

    # log results
    res_epoch = {
        'Loss': ep_loss,
        'Accuracy': metrics['Accuracy'],
    }

    return res_epoch, preds, labels


# def main():
#     global model, device, preprocess, criterion  # disable when not using def main
if __name__ == '__main__':
    # training mode: 1, unseen, all
    training_mode = 'all_attacks'

    ''' Initialization '''
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')

    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    ''' Load Model '''

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18.resnet18(weights=weights, weight_class=ResNet18_Weights)

    preprocess = weights.transforms()

    # load model_checkpoint.pt
    model_checkpoint = torch.load(join(run_dir, 'model_checkpoint.pt'))
    # model.load_state_dict(model_checkpoint['model_state_dict'])
    # model.load_state_dict(model_checkpoint)

    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data (copied from train.py)'''
    if True:
        paths_genuine, paths_attacks = dataset.read_annotations('both')
        paths_all = pd.concat([paths_genuine, paths_attacks])

        person_ids = pd.unique(paths_all['id0'])

        if training_mode == 'all_attacks':
            cat_names = dataset.label_names
        num_classes = len(dataset.labels)

        ''' Split dataset according to training mode '''
        label_nums = list(dataset.label_nums.values())
        if training_mode == 'all_attacks':
            ''' Train on all attacks, test on all attacks '''

            paths_all['label'] = paths_all['label_num']  # 0..7
            '''
            Dataset contains 10 people
            Split into 8 for training, 1 for validation, 1 for testing
            Every person has the same number of samples, but not the same number of attack types
            '''
            # split subsets based on person ids
            val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
            train_ids = np.setdiff1d(person_ids, [val_id, test_id])

            # split train/val/test (based on person IDs)
            paths_train = paths_all[paths_all['id0'].isin(train_ids)]
            paths_val = paths_all[paths_all['id0'].isin([val_id])]
            paths_test = paths_all[paths_all['id0'].isin([test_id])]

            class_train = class_val = class_test = 'all'
        else:
            raise ValueError(f'Unknown training mode: {training_mode}')

        ''' Safety check '''
        unique_classes = pd.concat([paths_train, paths_val, paths_test])['label'].nunique()
        assert unique_classes == num_classes, \
            f'Number of unique classes in dataset does not match number of classes in model\n' \
            f'real: {unique_classes}, expected: {num_classes}'

        # shuffle order - useful when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1).reset_index(drop=True)

        # limit size for prototyping
        limit = 640  # -1 for no limit, 3200  # TODO dataset size limit, do not forget
        paths_train = paths_train[:limit]
        paths_val = paths_val[:limit]
        paths_test = paths_test[:limit]

        # dataset loaders
        loader_kwargs = {'num_workers': 4, 'batch_size': 64}  # todo: as arguments
        train_loader = dataset.RoseYoutuLoader(paths_train, **loader_kwargs)
        val_loader = dataset.RoseYoutuLoader(paths_val, **loader_kwargs)
        test_loader = dataset.RoseYoutuLoader(paths_test, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

        label_names = dataset.label_names


    # Print setup
    print_dict(config_dict)
    # print_dict(args_dict)  # todo parse args

    ''' Evaluation '''
    print('Training set')
    res_train, preds_train, labels_train = eval_loop(train_loader)
    print_dict(res_train)

    print('Validation set')
    res_val, preds_val, labels_val = eval_loop(val_loader)
    print_dict(res_val)

    print('Test set')
    res_test, preds_test, labels_test = eval_loop(test_loader)
    print_dict(res_test)

    ''' Explainability '''
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.layer4[-1]]

    img, label = next(iter(test_loader))
    img, label = img[0:1].to(device), label[0:1].to(device)

    grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(cat) for cat in range(8)]

    cams = []
    for t in targets:
        grayscale_cam = grad_cam(input_tensor=img, targets=[t])

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        img_np = img[0].cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        cams.append(visualization)

    import seaborn as sns
    sns.set_context('poster')
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    # plot original image in top-left
    plt.subplot(3, 3, 1)
    plt.imshow(img_np)
    plt.title('Original image')
    plt.axis('off')

    for i, c in enumerate(cams):
        plt.subplot(3, 3, i + 2)
        plt.imshow(visualization)
        plt.title(label_names[i])
        plt.axis('off')
        # remove margin around image
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.tight_layout()
    plt.show()

    # note: results look the same for all classes.
    # the targets may have a wrong format, so the most-predicted class is used for all of the visualizations

    # todo: understand grad_cam code to reimplement myself
