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
from torch.nn.functional import softmax
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import ResNet18_Weights

import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

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
from util import get_dict, print_dict, xor, keys_append, save_dict_json
from model_util import EarlyStopping
import resnet18

# run_dir = 'runs/2023-01-10_14-41-03'  # 'unseen_attack'
run_dir = 'runs/2023-01-10_15-12-22'  # 'all_attacks'

model = None
device = None
preprocess = None
criterion = None

''' Global variables '''
# -

''' Parsing Arguments '''
parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
# parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
# parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)',
#                     type=str, default=None)
# parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, default=None)
parser.add_argument('-r', '--run', help='model/dataset/settings to load', type=str, default=None)


def eval_loop(loader):
    len_loader = len(loader)
    ep_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for img, label in tqdm(loader, leave=False, mininterval=1., desc='Eval'):
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
    args = parser.parse_args()
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    print('Loading model and setup from:', run_dir)

    # todo: in eval allow to overwrite config with args
    # if args.batch_size is not None:
    #     config_dict['batch_size'] = args.batch_size
    # if args.num_workers is not None:
    #     config_dict['num_workers'] = args.num_workers
    # if args.mode is not None:
    #     config_dict['mode'] = args.mode
    # if args.dataset is not None:
    #     config_dict['dataset'] = args.dataset

    training_mode = config_dict['mode']  # 'all_attacks'
    dataset_name = config_dict['dataset']  # 'rose_youtu'
    # load dataset module
    if dataset_name == 'rose_youtu':
        import dataset_rose_youtu as dataset_module
    elif dataset_name == 'siwm':
        import dataset_siwm as dataset_module
    else:
        raise ValueError(f'Unknown dataset name {dataset_name}')

    ''' Initialization '''
    print(f"Available GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    ''' Load Model '''
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18.resnet18(weights=weights, weight_class=ResNet18_Weights)
    model.fc = torch.nn.Linear(512, config_dict['num_classes'], bias=True)
    preprocess = weights.transforms()
    model.load_state_dict(torch.load(join(run_dir, 'model_checkpoint.pt')))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_dataset

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': config_dict['batch_size'],
                         'num_workers': config_dict['num_workers'], 'pin_memory': True}
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, limit=-1, quiet=False, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

    if False:
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

    if True:
        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)

        target_layers = [model.layer4[-1]]

        imgs, labels = next(iter(train_loader))
        preds_raw = model.forward(imgs.to(device))
        preds = softmax(preds_raw, dim=1).cpu().detach().numpy()

        # attempt to get a correct prediction
        while preds[i].argmax() != labels[i]:
            try:
                i += 1
                img = imgs[i:i + 1]
                label = labels[i:i + 1]
                pred = preds[i]
            except Exception as e:
                print(e)

        # for i, _ in enumerate(imgs):
        i = 0
        # img, label = imgs[i:i + 1], labels[i:i + 1]  # img 4D, label 1D
        label_scalar = label[0].item()  # label 0D
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)  # img_np 3D

        methods = [GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad]
        method_cams_dict = {}
        for method in tqdm(methods, desc='CAM methods', mininterval=1):
            method_name = method.__name__

            grad_cam = method(model=model, target_layers=target_layers, use_cuda=True)

            targets = [ClassifierOutputTarget(cat) for cat in range(config_dict['num_classes'])]

            cams = []
            overlayed = []
            for k, t in enumerate(targets):
                grayscale_cam = grad_cam(input_tensor=img, targets=[t])  # img 4D

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, ...]  # -> 3D

                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                cams.append(grayscale_cam)
                overlayed.append(visualization)

            method_cams_dict[method_name] = {'cams': cams, 'overlayed': overlayed}

            if False:
                ''' Plot CAMs '''
                sns.set_context('poster')
                fig, axs = plt.subplots(3, 3, figsize=(20, 20))
                plt.subplot(3, 3, 1)
                plt.imshow(img_np)
                plt.title('Original image')
                plt.axis('off')

                for j, c in enumerate(overlayed):
                    plt.subplot(3, 3, j + 2)
                    plt.imshow(c)
                    label_pred_score = f': {preds[i, j]:.2f}'
                    matches_label = f' (GT)' if j == label else ''
                    plt.title(dataset_module.label_names[j] + label_pred_score + matches_label)
                    plt.axis('off')
                    # remove margin around image
                    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                plt.tight_layout()
                # plt.show()

                # save figure fig to path
                path = join(cam_dir, f'{method_name}_{dataset_name}_img{i}_gt{label_scalar}.png')
                fig.savefig(path, bbox_inches='tight', pad_inches=0)

        # end of cam methods loop

        ''' Plot CAMs '''
        sns.set_context('poster')
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        plt.subplot(3, 3, 1)
        plt.imshow(img_np)
        plt.title(f'Original image')
        plt.axis('off')

        gt_label_name = dataset.label_names[label_scalar]
        pred_label_name = dataset.label_names[pred.argmax()]

        j = 0
        for name, cs in method_cams_dict.items():
            c = cs['overlayed'][label_scalar]
            plt.subplot(3, 3, j + 2)
            plt.imshow(c)
            label_pred_score = f': {preds[i, j]:.2f}'
            matches_label = f' (GT)' if j == label_scalar else ''
            plt.title(name)
            plt.axis('off')
            # remove margin around image
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            j += 1

        plt.suptitle(f'CAM Methods Comparison, "{gt_label_name}" class')
        plt.tight_layout()

        # save figure fig to path
        path = join(cam_dir, f'cam-comparison-gt{label_scalar}-rose_youtu.pdf')
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # labels names and order might be mixed up.
