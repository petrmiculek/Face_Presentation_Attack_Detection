# stdlib
import argparse
import json
import logging
import os
import datetime
from os.path import join
from copy import deepcopy

# fix for local import problems - add all local directories
import sys

sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# external
import torch

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

os.environ["WANDB_SILENT"] = "true"

import wandb as wb

from PIL import Image
import numpy as np
import os
import json

import torch
import torch.nn.functional as F

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
# import dataset_rose_youtu as dataset
from metrics import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, xor, keys_append, save_dict_json
from model_util import load_model
import resnet18

run_dir = ''
# run_dir = 'runs/2023-01-10_14-41-03'  # 'unseen_attack'
# run_dir = 'runs/2023-01-10_15-12-22'  # 'all_attacks'

# run_dir = 'runs/wandering-breeze-87'  # 'all_attacks', efficientnet_v2_s
# run_dir = 'runs/astral-paper-14'  # 'all_attacks', efficientnet_v2_s
# run_dir = 'runs/colorful-breeze-45'  # 'all_attacks', resnet18

model = None
device = None
preprocess = None
criterion = None

''' Global variables '''
# -

''' Parsing Arguments '''
parser = argparse.ArgumentParser()  # description='Evaluate model on dataset, run explanation methods'
# parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
# parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
# parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)',
#                     type=str, default=None)
# parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, default=None)
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)


def save_i(path, file, overwrite=False):
    """ Save but don't overwrite """
    exists = os.path.exists(path)
    if exists and not overwrite:
        print(f'File {path} exists, skipping saving.')
    else:
        if exists:
            print(f'File {path} exists, overwriting.')
        np.save(path, file)


def eval_loop(loader):
    """ Evaluate model on dataset """
    len_loader = len(loader)
    ep_loss = 0.0
    preds = []
    labels = []
    paths = []
    with torch.no_grad():
        for sample in tqdm(loader, mininterval=1., desc='Eval'):
            img, label = sample['image'], sample['label']
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

            paths.append(sample['path'])

    # loss is averaged over batch already, divide by batch number
    ep_loss /= len_loader

    metrics = compute_metrics(labels, preds)

    # log results
    res_epoch = {
        'Loss': ep_loss,
        'Accuracy': metrics['Accuracy'],
    }

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)

    return res_epoch, preds, labels, paths


def get_preds(dataset_loader, split_name, output_dir):
    path_save_preds = join(output_dir, f'preds_{split_name}.npy')
    path_save_labels = join(output_dir, f'labels_{split_name}.npy')
    path_save_paths = join(output_dir, f'paths_{split_name}.npy')

    if not os.path.isfile(path_save_preds) \
            or not os.path.isfile(path_save_paths) \
            or not os.path.isfile(path_save_labels):
        print(f'{split_name} set')
        res, preds, labels, paths = eval_loop(dataset_loader)
        print_dict(res)
        # save predictions to file
        save_i(path_save_preds, preds)
        save_i(path_save_labels, labels)
        save_i(path_save_paths, paths)
    else:
        # load predictions from file
        preds = np.load(path_save_preds)
        labels = np.load(path_save_labels)
        paths = np.load(path_save_paths)

    return {'labels': labels, 'preds': preds, 'paths': paths}


# def main():
#     global model, device, preprocess, criterion  # disable when not using def main
if __name__ == '__main__':
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    output_dir = join(run_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)

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
    # print(f"Available GPUs: {torch.cuda.device_count()}")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # print(f'Running on device: {device}')
    # print(f"Current device: {torch.cuda.current_device()}")
    # print(f"Device name: {torch.cuda.get_device_name(0)}")

    ''' Load Model '''
    model_name = config_dict['model_name']
    model, preprocess = load_model(model_name, config_dict['num_classes'])
    model.load_state_dict(torch.load(join(run_dir, 'model_checkpoint.pt'), map_location=device), strict=False)
    model.to(device)
    model.eval()

    # sample model prediction
    out = model(torch.rand(1, 3, 224, 224).to(device)).shape
    # assert shape is (1, num_classes)
    assert out == (1, config_dict['num_classes']), f'Model output shape is {out}'

    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_dataset

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        batch_size = 2  # 4 # 16  # 32  # config_dict['batch_size']  # depends on model and GPU memory
        num_workers = 4  # config_dict['num_workers']

        limit = -1  # -1 for full dataset

        loader_kwargs = {'shuffle': True, 'batch_size': batch_size,
                         'num_workers': num_workers, 'pin_memory': True}
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, limit=limit, quiet=False, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names

    ''' Evaluation '''
    if False:
        print('Training set')
        outputs_train = get_preds(train_loader, 'train', output_dir)

        print('Validation set')
        outputs_val = get_preds(val_loader, 'val', output_dir)

        print('Test set')
        outputs_test = get_preds(test_loader, 'test', output_dir)

        metrics_test = compute_metrics(outputs_test['labels'], outputs_test['preds'])
        metrics_test = keys_append(metrics_test, ' Test')

        cr = classification_report(outputs_test['labels'], outputs_test['preds'])
        print(cr)

        ''' Confusion matrix '''
        label_names = dataset_module.label_names_unified
        if True:
            cm_location = join(output_dir, 'confusion_matrix.pdf')
            confusion_matrix(outputs_test['labels'], outputs_test['preds'], output_location=cm_location,
                             labels=label_names, show=True)

            cm_binary_location = join(output_dir, 'confusion_matrix_binary.pdf')
            label_names_binary = ['genuine', 'attack']
            preds_binary = outputs_test['preds'] != bona_fide
            labels_binary = outputs_test['labels'] != bona_fide

            confusion_matrix(labels_binary, preds_binary, output_location=cm_binary_location, labels=label_names_binary,
                             show=True)




''' Explainability - GradCAM-like '''
if False:
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
        FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam_dir = join(run_dir, 'cam')
    os.makedirs(cam_dir, exist_ok=True)

    target_layers = [model.layer4[-1]]

    sample = next(iter(train_loader))
    imgs, labels = sample['image'], sample['label']
    with torch.no_grad():
        preds_raw = model.forward(imgs.to(device)).cpu()
        preds = softmax(preds_raw, dim=1).numpy()
        preds_classes = preds.argmax(axis=1)

    # for i, _ in enumerate(imgs):
    i = 0
    # attempt to get a correct prediction
    while preds[i].argmax() != labels[i]:
        try:
            i += 1
            img = imgs[i:i + 1]
            label = labels[i:i + 1]
            pred = preds[i]
        except Exception as e:
            print(e)
    print(f'Using image {i} with label {label} and prediction {pred}')

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

    gt_label_name = dataset_module.label_names[label_scalar]
    pred_label_name = dataset_module.label_names[pred.argmax()]

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

''' LIME '''
if False:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries


    def convert_for_lime(img):
        return (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


    def pred_hwc_np(images):
        """
        Run prediction on an image
        uses global model and device

        - check for batch size > 1  #DONE#
        :param images: HWC numpy array
        :return:
        """
        img0 = torch.tensor(images)
        if len(img0.shape) == 3:
            img0 = img0.unsqueeze(0)

        img0 = img0.permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(img0)
            probs = F.softmax(logits, dim=1)
            res = probs.cpu().numpy()
        return res


    explainer = lime_image.LimeImageExplainer()

    sample = next(iter(train_loader))
    imgs, labels = sample['image'], sample['label']
    with torch.no_grad():
        preds_raw = model.forward(imgs.to(device)).cpu()
        preds = softmax(preds_raw, dim=1).numpy()

    img_for_lime = (imgs[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # todo use convert_for_lime function
    label = labels[0].item()
    explanation = explainer.explain_instance(img_for_lime, pred_hwc_np,
                                             top_labels=5, hide_color=0, num_samples=1000)

    pred_top1 = explanation.top_labels[0]

    nums_to_names = dataset_module.nums_to_names
    pred_top1_name = nums_to_names[pred_top1]
    label_name = dataset_module.label_names[label]

    # positive-only
    temp, mask = explanation.get_image_and_mask(pred_top1, positive_only=True, num_features=5,
                                                hide_rest=True)
    img_lime_pos = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_lime_pos)
    plt.title(f'LIME explanation (pos), pred: {pred_top1_name}, GT: {label_name}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # positive and negative
    temp, mask = explanation.get_image_and_mask(pred_top1, positive_only=False, num_features=10,
                                                hide_rest=False)
    img_lime_posneg = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_lime_posneg)
    plt.title(f'LIME explanation (pos+neg), pred: {pred_top1_name}, GT: {label_name}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    """ Plot images """
    n_images = imgs.shape[0]
    fig, axs = plt.subplots(1, n_images, figsize=(10, 6))
    for i in range(n_images):
        label = labels[i].item()
        label_str = dataset_module.label_names[label]
        pred_class = preds_classes[i]
        pred_class_str = dataset_module.label_names[pred_class]
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        axs[i].imshow(img)
        axs[i].set_title(f'GT: {label_str}' + f', pred: {pred_class_str}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

    # weird - no GT "Video mac" + pred: "Printed still" is in the confusion matrix

''' Embeddings '''
if False:
    if not hasattr(model, 'fw_with_emb'):
        raise AttributeError('Model does not have fw_with_emb method')  # currently only for resnet18

    sample = next(iter(train_loader))
    imgs, labels, paths = sample['image'], sample['label'], sample['path']
    with torch.no_grad():
        preds, features = model.fw_with_emb(imgs.to(device))
        preds = preds.cpu().numpy()
        features = features.cpu().numpy()

    # embeddings distance matrix
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import euclidean_distances

    # cosine similarity
    cos_sim = cosine_similarity(features)
    # euclidean distance
    eucl_dist = euclidean_distances(features)
