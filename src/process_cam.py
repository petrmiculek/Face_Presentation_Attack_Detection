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
import warnings
from contextlib import suppress

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
import matplotlib

matplotlib.use('tkagg')
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
# show_plots = False
args = None

parser = argparse.ArgumentParser()  # description='Process CAMs'
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
parser.add_argument('-z', '--show', help='show outputs', action='store_true')
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)


def plot1x5(cams, title='', output_path=None):
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
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f'Saved to {output_path}, {title} (plot1x5)')
        if args.show:
            plt.show()
        plt.close(fig)


def plot2x3(cam_entry, img, output_path=None):
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
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f'Saved to {output_path}, {title} (plot2x3)')
    if args.show:
        plt.show()
    plt.close(fig)


def plot5x5(cams, output_path=None):
    """Plot 5x5 figure with a confusion matrix of CAMs."""
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    title = 'Average GradCAM Confusion Matrix'
    fig.suptitle(title)
    for i, class_pred in enumerate(label_names):  # rows == pred
        for j, class_gt in enumerate(label_names):  # cols == gt
            if i == 0:  # top-row: add title
                axs[i, j].set_title(class_gt)
            if j == 0:  # left-column: add ylabel
                axs[i, j].set_ylabel(class_pred, rotation=0, labelpad=30)

            axs[i, j].imshow(cams[i, j], vmin=0, vmax=255)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # remove frame around image
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f'Saved to {output_path}, {title} (plot5x5)')
    if args.show:
        plt.show()
    plt.close(fig)


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
    Process CAMs - perform analysis, visualize, and save plots.
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    run_dir = args.run
    path_cams = join(run_dir, 'cam', 'cams.pkl.gz')
    cams_id = 'gradcam'

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
    df = pd.read_pickle(path_cams)
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
            output_path = join(run_dir, 'cam', f'{cams_id}-img-{s["idx"]}.png')
            plot2x3(cam_entry, s['image'], output_path)

    ''' Average CAM by predicted category '''
    if True:
        cams_pred = np.zeros(cam_shape)  # (c, h, w)
        # filter all RuntimeWarning
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                cams_pred[i] = np.mean(cams[preds == i], axis=0)[i]
        cams_pred[np.isnan(cams_pred)] = 0

        output_path = join(run_dir, 'cam', f'{cams_id}-avg-class-pred.png')
        plot1x5(cams_pred, 'Average GradCAM per predicted class', output_path)

    ''' Average CAM by ground-truth category '''
    if True:
        cams_gt = np.zeros(cam_shape)  # (c, h, w)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                cams_gt[i] = np.mean(cams[labels == i], axis=0)[i]
        cams_gt[np.isnan(cams_gt)] = 0

        output_path = join(run_dir, 'cam', f'{cams_id}-avg-class-label.png')
        plot1x5(cams_gt, 'Average GradCAM per ground-truth class', output_path)

    ''' Average CAM per (predicted, ground-truth) category '''
    if True:
        # confusion matrix for CAMs
        cams_confmat = np.zeros((len(label_names), *cam_shape))  # (pred, label, h, w)
        cams = np.stack(df['cam'].values)
        preds = df['pred'].values
        labels = df['label'].values

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                for j in range(len(label_names)):
                    cams_confmat[i, j] = np.mean(cams[(preds == i) & (labels == j)], axis=0)[i]

        cams_confmat[np.isnan(cams_confmat)] = 0

        output_path = join(run_dir, 'cam', f'{cams_id}-avg-confmat.png')

        plot5x5(cams_confmat, output_path)

    ''' Incorrect predictions: image, predicted and ground truth CAMs '''
    if True:
        print(f'Incorrect predictions: {len(df[df["pred"] != df["label"]])} / {len(df)}')

        for i, row in df.iterrows():
            if row['pred'] == row['label']:
                continue

            img = ds[i]['image']
            pred = row['pred']
            label = row['label']
            cam_pred = row['cam'][pred]
            cam_label = row['cam'][label]
            idx = row['idx']

            ''' Overlay CAM (plot1x3) '''
            overlayed_pred = overlay_cam(img, cam_pred)
            overlayed_label = overlay_cam(img, cam_label)

            output_path = join(run_dir, 'cam', f'{cams_id}-incorrect-pred-{idx}.png')
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

            if output_path is not None:
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                print(f'Saved to {output_path}, {title} (plot1x3)')

            if args.show:
                plt.show()
            plt.close(fig)

    ''' Further possible comparisons '''
    ''' Difference between predicted and ground truth category '''
    # not now
    ''' Difference between predicted and ground truth category, per class '''
    # not now
