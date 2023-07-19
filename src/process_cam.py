"""
cams.pkl.gz:
    cams
    idx
    pred
    label
    TODO make format complete [clean]

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
from glob import glob
from os.path import join, basename, isfile, dirname
import warnings

# fix for local import problems - add all local directories
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# external

os.environ["WANDB_SILENT"] = "true"

import numpy as np
import pandas as pd
import torch
import matplotlib
import seaborn as sns
from tqdm import tqdm

if "PYCHARM_HOSTED" in os.environ:
    matplotlib.use('module://backend_interagg')  # default backend for Pycharm
elif "SCRATCHDIR" in os.environ:
    pass
else:
    matplotlib.use('tkagg')  # works in console
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
from dataset_base import show_labels_distribution
from util_image import overlay_cam, get_marker
from util_torch import init_seed, get_dataset_module

''' Global variables '''
run_dir = ''
nums_to_names = None
label_names = None
# show_plots = False
args = None
cam_dir = None
ext = 'png'  # extension for saving plots

parser = argparse.ArgumentParser()  # description='Process CAMs'
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
parser.add_argument('-z', '--show', help='show outputs', action='store_true')
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)
parser.add_argument('-n', '--no_log', help='do not save anything', action='store_true')
parser.add_argument('-p', '--path_prefix', help='path to dataset')
parser.add_argument('-f', '--files', help='path(s) to CAMs', type=str, nargs='+')


def read_cams(paths):
    """ Read pickled CAM DataFrame/s. """
    df_ = None
    if isinstance(paths, str):
        df_ = pd.read_pickle(paths)
    elif isinstance(paths, list):
        # merge dataframes
        lens = []
        for p in tqdm(paths):
            try:
                df_tmp = pd.read_pickle(p)
                source = basename(p)
                df_tmp['path'] = df_tmp['path'].apply(basename)
                df_tmp['source'] = source  # broadcast to all rows
                print(f'Loading {source}, shape: {df_tmp.shape}')
                if df_ is None:
                    df_ = df_tmp
                    lens.append(len(df_tmp))
                    continue
                ''' Check for overlapping lines by: idx, method '''
                intersection = df_tmp.merge(df_, on=['idx', 'method'], how='inner')
                if not intersection.empty:
                    print(f'Warning: {source} intersects with: {intersection["file_y"].unique()}')
                ''' Check for conflicting attributes - label, pred '''
                conflicting = (intersection['label_x'] != intersection['label_y']) | \
                              (intersection['pred_x'] != intersection['pred_y'])
                if any(conflicting):
                    print(f'Warning: {source} and {intersection["file"].iloc[0]} have conflicting labels/preds')
                lens.append(len(df_tmp))

                df_ = pd.concat([df_, df_tmp], ignore_index=True)
            except Exception as e:
                print(e)
                continue

        print(
            f'Merged {len(lens)}/{len(paths)} dataframes, total shape: {df_.shape}, original lengths: {lens} -> {sum(lens)}')

    return df_


def plot1x5(cams, title='', output_path=None):
    """Plot 5 CAMs in a row."""
    if True:
        fig, axs = plt.subplots(1, 5, figsize=(16, 4), sharex=True)
        for jc, ax in zip(enumerate(cams), axs.flat):
            j, c = jc
            if c.shape[0] == 1:
                c = c[0]
            # plt.subplot(1, 5, j + 1)
            im = ax.imshow(c, vmin=0, vmax=255)
            # make sure the aspect ratio is correct
            ax.set_aspect('equal')
            ax.set_title(label_names[j])
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=-0.05)
        fig.suptitle(title)
        if output_path is not None and not args.no_log:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f'Saved to {output_path}, {title} (plot1x5)')
        if args.show:
            plt.show()
        plt.close(fig)


def plot2x3(cam_entry, img, output_path=None, title='', **kwargs):
    """Plot 2x3 figure with image and 5 CAMs."""
    cam = cam_entry['cam']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    pred_name = nums_to_names[cam_entry['pred']]
    gt_name = nums_to_names[cam_entry["label"]]
    method = cam_entry['method']
    plt.suptitle(f'{method} {cam_entry["idx"]}, pred: {pred_name}, gt: {gt_name}')

    ''' Image '''
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')

    ''' CAMs '''
    for j, c in enumerate(cam):
        if c.ndim == 3 and c.shape[0] == 1:
            c = c[0]

        plt.subplot(2, 3, j + 2)
        plt.imshow(c, **kwargs)  # uint8 heatmaps, don't normalize
        plt.axis('off')
        ax_title = nums_to_names[j]
        if j == cam_entry['label']:
            ax_title += ' (GT)'
        if j == cam_entry['pred']:
            ax_title += ' (pred)'
        plt.title(ax_title)

    plt.tight_layout()
    if output_path is not None and not args.no_log:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f'Saved to {output_path}, {title} (plot2x3)')
    if args.show:
        plt.show()
    plt.close(fig)


def plot5x5(cams, output_path=None):
    """Plot 5x5 figure with a confusion matrix of CAMs.

    cams: Input matrix: [gt, pred, h, w]
    """
    global label_names
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    title = 'Average GradCAM Confusion Matrix'
    fig.suptitle(title, fontsize=16)

    for i, class_gt in enumerate(label_names):  # rows == gt
        for j, class_pred in enumerate(label_names):  # cols == pred
            if i == 0:  # top-row
                axs[i, j].set_title(class_pred, fontsize=12)
                print(class_pred, end=' ')
            if j == 0:  # left-column
                axs[i, j].set_ylabel(class_gt, rotation=90, labelpad=5, fontsize=12)
                print(class_gt, end=' ')
            c = cams[i, j]
            if c.shape[0] == 1:
                c = c[0]
            axs[i, j].imshow(c, vmin=0, vmax=255)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # remove frame around image
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    # write at the left: "predicted"
    axs[2, 0].text(-0.2, 0.5, 'predicted', rotation=90, va='center', ha='center', transform=axs[2, 0].transAxes,
                   fontsize=14)

    if output_path is not None and not args.no_log:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f'Saved to {output_path}, {title} (plot5x5)')
    if args.show:
        plt.show()
    plt.close(fig)


def deletion_per_image(df, many_ok=False):
    global args, cam_dir, ext, percentages_kept
    # x: perturbation level
    # y: prediction drop
    if len(df) > 200 and not many_ok:
        print(f'deletion_per_image: too many {len(df)} samples, '
              f'override by `many_ok` to run anyway.')

    y = df['del_scores'].values
    idxs = df['idx'].values
    plt.figure(figsize=(6, 4))
    for i, scores_line in enumerate(y):
        sample_idx = idxs[i]
        marker_random = get_marker(sample_idx)
        plt.plot(percentages_kept, scores_line, label=sample_idx, marker=marker_random)
    plt.xticks(percentages_kept)  # original x-values ticks
    plt.gca().invert_xaxis()  # decreasing x axis
    plt.ylim(0, 1.05)
    # plt.legend(title='Sample ID')  # too long for per-sample legend
    plt.ylabel('Prediction Score')
    plt.xlabel('Perturbation Intensity (% of image kept)')
    plt.title('Deletion Metric')
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    output_path = join(cam_dir, f'deletion_metric_samples.{ext}')
    if output_path and not args.no_log:
        plt.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
    return percentages_kept


# def main():
if __name__ == '__main__':
    """
    Process CAMs - perform analysis, visualize, and save plots.
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))

    ''' Process paths and filenames '''
    # paths like: 'runs/sage-cherry-91/eval-rose_youtu-single-all_attacks-all/cam/cams_SobelFakeCAM_20.pkl.gz'
    if "PYCHARM_HOSTED" in os.environ:
        # glob expand file argument
        args.files = glob(args.files[0])

    ''' Directories/paths '''
    cam_dir = dirname(args.files[0])
    path_list = args.files[0].split('/')  # assume common directory prefix for cam files
    run_dir = join(*path_list[:2])  # rebuild runs/<run_name>

    eval_dir = path_list[2]

    # format to parse, hyphen-separated
    # 'eval', dataset_name, dataset_note [, attack], mode, limit_
    _, dataset_name, dataset_note, mode, *attack, limit_ = eval_dir.split('-')
    attack = attack[0] if len(attack) == 1 else None

    ''' Parse filenames '''
    # filenames example: cams_EigenCAM_20.pkl.gz
    # filenames pattern: cams_<method>_<limit>.pkl.gz

    # read all limit strings from filenames
    filenames = [basename(f) for f in args.files]
    limits = set(f.split('_')[2].split('.')[0] for f in filenames)
    if len(limits) != 1:
        raise ValueError(f'Multiple different limits in filenames: {limits}')

    methods = [f.split('_')[1] for f in filenames]
    limit = limits.pop()  # only one in the set (which can also be destroyed)
    limit = -1 if limit == 'all' else int(limit)

    ''' Load CAMs '''
    df = read_cams(args.files)

    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    print('Loading model and setup from:', run_dir)

    ''' Arguments '''
    # training_mode = config_dict['mode']  # 'all_attacks'
    # dataset_name = config_dict['dataset']  # 'rose_youtu'
    dataset_module = get_dataset_module(dataset_name)

    ''' Initialization '''
    seed = args.seed if args.seed is not None else config.seed_eval_default  # 42
    init_seed(seed)

    # plotting params
    sns.set_style('whitegrid')  # seaborn set style

    ''' Load Dataset '''
    if True:
        from dataset_base import pick_dataset_version, load_annotations

        dataset_meta = pick_dataset_version(dataset_name, mode, attack, dataset_note)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']
        split = 'test'  # implicitly always working with unseen data
        dataset_paths = load_annotations(dataset_meta, seed, args.path_prefix, limit)[split]
        ds = dataset_module.Dataset(dataset_paths)
        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified
        num_classes = dataset_meta['num_classes']
        ''' Show labels distribution '''
        show_labels_distribution(dataset_paths['label'], split, num_classes)

    t0 = time.perf_counter()

    ###############################################################################
    ''' Exploring CAMs '''
    nums_to_names = dataset_module.nums_to_unified
    label_names = dataset_module.label_names_unified
    idxs = df['idx'].values

    # -------------------------------------------- vvv
    # temporary: shapes of CAMs
    # cam_shape = (5, 1, 7, 7)  # temporary
    shp = lambda x: x.shape
    df['shp'] = df.cam.apply(shp)
    shape_value_counts = df.shp.value_counts()
    if len(df.shp.unique()) != 1:
        print('Warning: different CAM shapes!\n', shape_value_counts)
    cam_shape = df['cam'].values[0].shape
    dg = df[['method', 'shp']].groupby('shp')
    for g, rows in dg:
        print(g, rows.method.value_counts())

    df_orig = df.copy()
    df = df[df['shape'] == (5, 1, 7, 7)]
    # -------------------------------------------- ^^^
    print(f'CAM shape: {cam_shape}')
    cams = np.stack(df['cam'].values)
    labels = df['label'].values
    preds = df['pred'].values
    percentages_kept = df['percentages_kept'].values[0]  # assume same for all
    auc_percentages = np.array(percentages_kept) / 100  # normalize to [0, 1]

    ''' Across-method correlation '''
    if False:
        # todo `df` was filtered down, fyi
        from util_image import plot_many
        from scipy.stats import pearsonr, spearmanr

        # step 1: single sample
        idx = df.iloc[0].idx
        ds_idx = [sample['idx'] for sample in ds]
        sample = ds[ds_idx.index(idx)]
        img = sample['image']

        # step 2: get all cams for this sample
        df_sample = df[df.idx == idx]
        methods = df_sample.method.values
        cams = np.stack(df_sample.cam.values)
        cams = cams[:, :, 0, ...]  # drop 1-channel dim  -- not generally applicable
        cam1 = cams[0]
        cam2 = cams[1]
        # plot_many([img, cam1, cam2], titles=['img', methods[0], methods[1]])

        for i, c in enumerate(label_names):
            print(f'{i}: {c}')
            c1 = cam1[i]
            c2 = cam2[i]

            # if constant (all 0), corrcoef = 0
            if np.allclose(c1, 0) or np.allclose(c2, 0):
                corr_pearson = 0
                corr_spearman = 0
            else:
                corrcoef = np.corrcoef(c1.flatten(), c2.flatten())  # -> [2, 2]
                corr_pearson = pearsonr(c1.flatten(), c2.flatten())
                corr_spearman = spearmanr(c1.flatten(), c2.flatten())

            print(f'{c}: p={corr_pearson:.3f} s={corr_spearman:.3f}')
            plot_many([c1, c2], title=f'{c}: p={corr_pearson:.3f} s={corr_spearman:.3f}',
                      titles=[methods[0], methods[1]])

        """
        .
        """

    ''' Per-image Deletion Metric Plot'''
    if False:
        # Not used, since per-sample visualization is not practical for thousands of samples.
        percentages_kept = deletion_per_image(df)

    ''' Per-Method Deletion Metric Plot '''
    if True:
        methods_unique = df.method.unique()
        for m in methods_unique:
            df_method = df[df.method == m]
            del_scores = np.stack(df_method.del_scores.values)
            ys = np.mean(del_scores, axis=0)
            stds = np.std(del_scores, axis=0)
            auc = np.trapz(ys, -auc_percentages)
            plt.plot(percentages_kept, ys, '.-', label=f'{m}: {auc:.3f}')
            plt.fill_between(percentages_kept, ys - stds, ys + stds, alpha=0.2)

        plt.title('Deletion Metric per Method')
        plt.xlabel('% pixels kept')
        plt.ylabel('Confidence')
        plt.xticks(percentages_kept)  # original x-values ticks
        plt.gca().invert_xaxis()  # decreasing x axis
        plt.ylim(0, 1.05)
        plt.legend(title='Method: AUC')
        plt.tight_layout()
        output_path = join(cam_dir, f'deletion_metric_methods.{ext}')
        if output_path is not None and not args.no_log:
            plt.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

    ''' Per-class Deletion Metric Plot'''
    if False and 'baseline' in df:
        """ 
        Single method, subplot per baseline, plotting classes separately. 
        Compare ways of perturbing the image with different baselines.
        """
        baselines_unique = df['baseline'].unique()

        aucs = []
        for baseline in baselines_unique:
            # x: perturbation level
            # y: prediction drop
            # hue: class
            # baseline = df['baseline'].values[0]
            df_base = df[df.baseline == baseline]
            y = df_base['del_scores'].values
            preds = df_base['pred'].values
            empty = []
            for i, label in enumerate(label_names):
                y_pred_class = y[preds == i]
                if len(y_pred_class) == 0:
                    empty.append(label)
                    continue

                y_pred_class = np.stack(y_pred_class)
                mean_y = np.mean(y_pred_class, axis=0)
                std_y = np.std(y_pred_class, axis=0)
                auc = np.trapz(mean_y, -auc_percentages)  # -xs because decreasing x-axis, but auc() assumes increasing
                aucs.append({'baseline': baseline, 'class': label, 'auc': auc})
                # plot mean + std
                plt.plot(percentages_kept, mean_y, label=f'{label}: {auc:.3f}')
                plt.fill_between(percentages_kept, mean_y - std_y, mean_y + std_y, alpha=0.2)

            if len(empty):
                print(f'Empty classes: {empty}')

            plt.xticks(percentages_kept)  # original x-values ticks
            plt.gca().invert_xaxis()  # decreasing x axis
            plt.ylim(0, 1.05)
            plt.legend(title='Class')
            plt.ylabel('Prediction Score')
            plt.xlabel('Perturbation Intensity (% of image kept)')
            plt.title(f'Deletion Metric ({baseline}) by Class')

            # remove top and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.tight_layout()
            output_path = join(cam_dir, f'deletion_metric_classes.{ext}')
            if output_path and not args.no_log:
                plt.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
            if args.show:
                plt.show()
            plt.close()

        aucs_df = pd.DataFrame(aucs)
        average_auc = aucs_df.groupby('baseline')['auc'].mean()
        print(average_auc.sort_values(ascending=True))

    ''' Per-image CAM - per-class heatmaps '''
    if False:
        for i in range(3):
            s = ds[i]
            idx = s['idx']
            entries = df[df.idx == idx]
            if entries.empty:
                # when loading more methods, idx is not unique
                print(f"dataset index {s['idx']} not found in df")
                continue

            for _, cam_entry in entries.iterrows():
                cam_method_name = cam_entry['method']
                output_path = join(cam_dir, f'{cam_method_name}_{limit}_idx{idx}.{ext}')
                plot2x3(cam_entry, s['image'], output_path, **{'vmin': 0, 'vmax': 255, 'cmap': 'viridis'})

    ''' Average CAM by predicted category '''
    if False:
        cams_pred = np.zeros(cam_shape)  # (c, h, w)
        # filter all RuntimeWarning
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                cams_pred[i] = np.mean(cams[preds == i], axis=0)[i]
        cams_pred[np.isnan(cams_pred)] = 0

        output_path = join(cam_dir, f'{cams_id}_avg_class_pred.{ext}')
        plot1x5(cams_pred, 'Average CAM per predicted class', output_path)

    ''' Average CAM by ground-truth category '''
    if False:
        cams_gt = np.zeros(cam_shape)  # (c, h, w)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                cams_gt[i] = np.mean(cams[labels == i], axis=0)[i]
        cams_gt[np.isnan(cams_gt)] = 0

        output_path = join(cam_dir, f'avg_class_label.{ext}')
        plot1x5(cams_gt, 'Average CAM per ground_truth class', output_path)

    ''' Average CAM per (predicted, ground-truth) category '''
    if False:
        # confusion matrix for CAMs
        cams_confmat = np.zeros((len(label_names), *cam_shape))  # (pred, label, h, w)

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):  # i == label (gt)
                for j in range(len(label_names)):  # j == pred
                    cams_confmat[i, j] = np.mean(cams[(labels == i) & (preds == j)], axis=0)[j]

        cams_confmat[np.isnan(cams_confmat)] = 0
        output_path = join(cam_dir, f'avg_confmat.{ext}')
        plot5x5(cams_confmat, output_path)

    ''' Incorrect predictions: image, predicted and ground truth CAMs '''
    n_incorrect = len(df[df["pred"] != df["label"]])
    if False and n_incorrect > 0:
        print(f'Incorrect predictions: {n_incorrect} / {len(df)}')

        for i, row in df.iterrows():
            if row['pred'] == row['label']:
                continue
            idx = row['idx']
            s = ds[idx]
            img = s['image']
            pred = row['pred']
            label = row['label']
            cam_pred = row['cam'][pred]
            cam_label = row['cam'][label]

            ''' Overlay CAM (plot1x3) '''
            overlayed_pred = overlay_cam(img, cam_pred)
            overlayed_label = overlay_cam(img, cam_label)

            output_path = join(cam_dir, f'{cams_id}_incorrect_pred_{idx}.{ext}')
            title = f'Prediction Error'

            # 1x3 figure: image, predicted cam, ground truth cam
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].set_title(f'Image[{idx}]')
            axs[0].axis('off')
            axs[1].imshow(overlayed_pred)
            axs[1].set_title(f'Predicted: {nums_to_names[pred]}')
            axs[1].axis('off')
            axs[2].imshow(overlayed_label)
            axs[2].set_title(f'Ground truth: {nums_to_names[label]}')
            axs[2].axis('off')
            plt.suptitle(title)
            plt.tight_layout()

            if output_path is not None and not args.no_log:
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
