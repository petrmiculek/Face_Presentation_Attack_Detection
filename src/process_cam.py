"""
cams.pkl.gz:
    cams
    idx
    pred
    label

I need:
model dir (runs/name), but not the model itself
dataset - bare, to index
"""

# stdlib
import argparse
import logging
import os
import json
import sys
import time
from os.path import join

# fix for local import problems - add all local directories
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# external
from sklearn.metrics import classification_report
from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
from metrics import compute_metrics, confusion_matrix
import config

''' Global variables '''
run_dir = ''
nums_to_names = None
label_names = None

parser = argparse.ArgumentParser()  # description='Process CAMs'
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
# parser.add_argument('-z', '--show', help='show outputs', action='store_true')
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)


def plot5(cams, title='', output_path=None):
    """Plot 5 CAMs in a row."""
    if True:
        fig, axs = plt.subplots(1, 5, figsize=(16, 4), sharex=True)  #
        for jc, ax in zip(enumerate(cams), axs.flat):
            j, c = jc
            # plt.subplot(1, 5, j + 1)
            im = ax.imshow(c, vmin=0, vmax=255)
            # make sure the aspect ratio is correct
            ax.set_aspect('equal')
            ax.set_title(label_names[j])
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=-0.05)
        fig.suptitle(title)
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()


def plot2x3(cam_entry, img, run_dir):
    """Plot 2x3 figure with image and 5 CAMs."""
    cam = cam_entry['cam']
    # figure 2x3
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    pred_name = nums_to_names[cam_entry['pred']]
    gt_name = nums_to_names[cam_entry["label"]]
    plt.suptitle(f'GradCAM {cam_entry["idx"]}, pred: {pred_name}, gt: {gt_name}')

    ''' Image '''
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')

    ''' CAMs '''
    for j, c in enumerate(cam):
        plt.subplot(2, 3, j + 2)
        plt.imshow(c, vmin=0, vmax=255)  # uint8 heatmaps, don't normalize
        plt.axis('off')
        title = nums_to_names[j]
        if j == cam_entry['label']:
            title += ' (GT)'
        if j == cam_entry['pred']:
            title += ' (pred)'
        plt.title(title)

    plt.tight_layout()
    output_path = join(run_dir, 'cam', f'cam-{cam_entry["idx"]}.png')
    plt.savefig(output_path)
    plt.show()
    plt.close(fig)


def plot5x5(canvas, output_path=None):
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Average GradCAM Confusion Matrix')
    for i, class_pred in enumerate(label_names):  # rows == pred
        for j, class_gt in enumerate(label_names):  # cols == gt
            if i == 0:  # top-row: add title
                axs[i, j].set_title(class_gt)
            if j == 0:  # left-column: add ylabel
                axs[i, j].set_ylabel(class_pred, rotation=0, labelpad=30)

            axs[i, j].imshow(canvas[i, j], vmin=0, vmax=255)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # remove frame around image
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.show()


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


# def main():
if __name__ == '__main__':
    """
    Process CAMs
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    output_dir = join(run_dir, 'cam')
    os.makedirs(output_dir, exist_ok=True)

    print('Loading model and setup from:', run_dir)

    ''' Arguments '''
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
    seed = args.seed if args.seed is not None else config.seed_eval_default  # 42
    print(f'Random seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True

    limit = -1 if args.limit is None else args.limit

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_annotations

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        split = 'test'
        dataset_paths = load_annotations(dataset_meta, seed, limit)[split]
        ds = dataset_module.Dataset(dataset_paths)

        # train_loader, val_loader, test_loader = \
        #     load_dataset(dataset_meta, dataset_module, limit=limit, quiet=False, **loader_kwargs)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified

    t0 = time.perf_counter()

    ''' Show dataset labels distribution '''
    if True:
        num_classes = dataset_meta['num_classes']
        print('Dataset labels per split:')  # including labels not present
        class_occurences = []
        value_counts = dataset_paths['label'].value_counts().sort_index()
        for i in range(num_classes):
            if i in value_counts.index:
                class_occurences.append(value_counts[i])
            else:
                class_occurences.append(0)

        print(f'{split}:', class_occurences)
    ####################################################################################################################
    ''' Exploring CAMs '''
    nums_to_names = dataset_module.nums_to_unified
    label_names = dataset_module.label_names_unified
    df = pd.read_pickle(join(run_dir, 'cam', 'cams-1000.pkl.gz'))
    idxs = df['idx'].values
    cam_shape = df['cam'].values[0].shape
    print(f'CAM shape: {cam_shape}')
    cams = np.stack(df['cam'].values)
    labels = df['label'].values
    preds = df['pred'].values

    ''' Per-image CAM for all classes '''
    if False:
        for i in range(3):
            s = ds[i]
            cam_entry = df.iloc[i]
            if s['idx'] != cam_entry['idx']:
                print(f"index mismatch: {s['idx']}, {cam_entry['idx']}")
                continue
            plot2x3(cam_entry, s['image'], run_dir)

    ''' Average CAM by predicted category '''
    if True:
        # # only take winning prediction, ignore soft probabilities
        # cam_sum = np.zeros(cam_shape)
        # n_preds = np.zeros(len(label_names))
        # for i, row in df.iterrows():
        #     # sum per-channel
        #     pred = row['pred']
        #     cam_sum[pred] += row['cam'][pred]
        #     n_preds[pred] += 1
        #
        # # divide by number of predictions (+1 div-zero, imprecision acceptable)
        # cam_avg = cam_sum // (n_preds[:, None, None] + 1)

        cams_pred = np.zeros(cam_shape)  # (c, h, w)
        for i in range(len(label_names)):
            cams_pred[i] = np.mean(cams[preds == i], axis=0)[i]
        cams_pred[np.isnan(cams_pred)] = 0

        output_path = join(run_dir, 'cam', f'avg-class-pred-{1000}.png')
        plot5(cams_pred, 'Average GradCAM per predicted class', output_path)

    ''' Average CAM by ground-truth category '''
    if True:
        ''' Version 1 - kept for reference '''
        # cam_sum = np.zeros(cam_shape)
        # n_labels = np.zeros(len(label_names))
        # for i, row in df.iterrows():
        #     # sum per-channel
        #     label = row['label']
        #     cam_sum[label] += row['cam'][label]
        #     n_labels[label] += 1
        #
        # # divide by number of predictions (+1 div-zero, imprecision acceptable)
        # cam_avg = cam_sum // (n_labels[:, None, None] + 1)

        ''' Version 2 '''
        cams_gt = np.zeros(cam_shape)  # (c, h, w)
        for i in range(len(label_names)):
            cams_gt[i] = np.mean(cams[labels == i], axis=0)[i]
        cams_gt[np.isnan(cams_gt)] = 0

        output_path = join(run_dir, 'cam', f'avg-class-label-{1000}.png')
        plot5(cams_gt, 'Average GradCAM per ground-truth class', output_path)

    ''' Average CAM per (predicted, ground-truth) category '''
    # one CAM for each confusion matrix cell
    if True:
        cams_confmat = np.zeros((len(label_names), *cam_shape))  # (pred, label, h, w)
        cams = np.stack(df['cam'].values)
        preds = df['pred'].values
        labels = df['label'].values

        for i in range(len(label_names)):
            for j in range(len(label_names)):
                cams_confmat[i, j] = np.mean(cams[(preds == i) & (labels == j)], axis=0)[i]

        cams_confmat[np.isnan(cams_confmat)] = 0

        output_path = join(run_dir, 'cam', f'avg-class-pred-label-{1000}.png')

        plot5x5(cams_confmat, output_path)

    ''' Incorrect predictions: image, predicted and ground truth CAMs '''
    if True:
        print('manually fail first 5 preds')
        for i, row in df.iterrows():
            if row['pred'] == row['label'] and i > 4:
                continue
            else:
                print(i, end=", ")

            img = ds[i]['image']
            pred = row['pred']
            label = row['label']
            cam_pred = row['cam'][pred]
            cam_label = row['cam'][label]
            idx = row['idx']

            ''' Overlay CAM '''
            overlayed_pred = overlay_cam(img, cam_pred)
            overlayed_label = overlay_cam(img, cam_label)
            # plt.imshow(overlayed_pred); plt.colorbar(); plt.axis('off'); plt.show()

            output_path = join(run_dir, 'cam', f'incorrect-pred-{idx}.png')
            title = f'{idx}: {nums_to_names[pred]} instead of {nums_to_names[label]}'

            # 1x3 figure: image, predicted cam, ground truth cam
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].set_title('Image')
            axs[0].axis('off')
            axs[1].imshow(overlayed_pred)
            axs[1].set_title('Predicted')
            axs[1].axis('off')
            axs[2].imshow(overlayed_label)
            axs[2].set_title('Ground truth')
            axs[2].axis('off')
            plt.suptitle(title)
            plt.tight_layout()
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.show()

    ''' Further possible comparisons '''
    ''' Difference between predicted and ground truth category '''
    # not now
    ''' Difference between predicted and ground truth category, per class '''
    # not now

    ''' Unused copied code of generating CAMs '''
    if False:
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            FullGrad
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import seaborn as sns

        methods = [GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad]

        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)

        # print list of model layers
        # for name, param in model.named_parameters():
        #     print(name, param.shape)

        target_layers = model.features  # [model.layer4[-1]]  # resnet18

        labels_list = []
        paths_list = []
        preds_list = []
        idxs_list = []
        cams_all = dict()

        nums_to_names = dataset_module.nums_to_unified

        ''' Iterate over batches in dataset '''
        for batch in tqdm(test_loader, mininterval=2., desc='CAM'):
            img_batch, label = batch['image'], batch['label']
            img_batch, label_batch = batch['image'], batch['label']
            path_batch = batch['path']
            with torch.no_grad():
                preds_raw = model(img_batch.to(device)).cpu()
                preds = F.softmax(preds_raw, dim=1).numpy()
                preds_classes = np.argmax(preds, axis=1)

            ''' Iterate over images in batch '''
            labels_list.append(label_batch)
            paths_list.append(path_batch)
            idxs_list.append(batch['idx'])
            for i, img in enumerate(img_batch):
                # tqdm(..., mininterval=2., desc='\tBatch', leave=False, total=len(img_batch)):

                pred = preds[i]
                idx = batch['idx'][i].item()
                label = label_batch[i].item()

                # img, label = img_batch[i:i + 1], label_batch[i:i + 1]  # img 4D, label 1D
                img_np = img.cpu().numpy().transpose(1, 2, 0)  # img_np 3D
                # img_np_batch = img_batch.cpu().numpy().transpose(0, 2, 3, 1)  # img_np 4D

                img_cams = {}

                for method in methods:  # tqdm(..., desc='CAM methods', mininterval=1, leave=False):
                    method_name = method.__name__
                    # explanations by class (same method)
                    cams = []
                    overlayed = []
                    for k, t in enumerate(targets):
                        grayscale_cam = grad_cam(input_tensor=img[None, ...], targets=[t])  # img 4D

                        # In this example grayscale_cam has only one image in the batch:
                        grayscale_cam = grayscale_cam[0, ...]  # -> 3D

                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                        cams.append(grayscale_cam)
                        overlayed.append(visualization)

                    img_cams[method_name] = {'cams': cams, 'overlayed': overlayed}

                    ''' Plot CAMs '''
                    if True:
                        # explanation by class (same method)
                        sns.set_context('poster')
                        fig, axs = plt.subplots(2, 3, figsize=(20, 16))
                        plt.subplot(2, 3, 1)
                        plt.imshow(img_np)
                        plt.title('Original image')
                        plt.axis('off')

                        for j, c in enumerate(overlayed):
                            plt.subplot(2, 3, j + 2)
                            plt.imshow(c)
                            label_pred_score = f': {preds[i, j]:.2f}'
                            matches_label = f' (GT)' if j == label else ''
                            plt.title(label_names[j] + label_pred_score + matches_label)
                            plt.axis('off')
                            # remove margin around image
                            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                        plt.tight_layout()

                        if args.show:
                            plt.show()

                        # save figure fig to path
                        path = join(cam_dir, f'{method_name}_{dataset_name}_img{idx}_gt{label}.png')
                        fig.savefig(path, bbox_inches='tight', pad_inches=0)

                        # close figure
                        plt.close(fig)

                    # end of cam methods loop

                cams_all[idx] = img_cams

                ''' Plot CAMs '''
                if True:
                    # explanation by method (predicted class)
                    sns.set_context('poster')
                    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
                    plt.subplot(3, 3, 1)
                    plt.imshow(img_np)
                    plt.title(f'Original image')
                    plt.axis('off')

                    gt_label_name = label_names[label]
                    pred_label_name = label_names[pred.argmax()]

                    j = 0
                    for name, cs in img_cams.items():
                        c = cs['overlayed'][label]
                        plt.subplot(3, 3, j + 2)
                        plt.imshow(c)
                        plt.title(name)
                        plt.axis('off')
                        # remove margin around image
                        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        j += 1

                    plt.suptitle(f'CAM Methods Comparison, GT: "{gt_label_name}" pred: {pred_label_name}')
                    plt.tight_layout()

                    # save figure fig to path
                    path = join(cam_dir, f'cam-comparison-gt{label}-rose_youtu.pdf')
                    fig.savefig(path, bbox_inches='tight', pad_inches=0)
                    if args.show:
                        plt.show()

                    plt.close(fig)

                # end of images in batch loop
            # end of batches in dataset loop
        # end of CAM methods section

        ''' Save CAMs npz '''
        ...
