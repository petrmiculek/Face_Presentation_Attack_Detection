#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'
"""
Perform analyses on CAMs
- Load and preprocess CAMs
- Make analyses
- Create plots


cams.pkl.gz:
['cam', 'idx', 'path', 'method', 'percentages_kept', 'baseline', 'label',
   'pred', 'del_scores', 'ins_scores', 'pred_scores', 'source']
joined with annotation paths:
['label_text', 'speaking', 'device', 'glasses', 'environment', 'id1', 'id2',
   'path_y', 'box_orig', 'box', 'landmarks', 'dim_orig', 'face_prob',
   'laplacian_var', 'label_orig', 'label_unif', 'label_bin', 'label_y',]
computed:
['shp', 'del_scores_pred', 'auc', 'cam_pred', 'auc_del', 'auc_ins']

Script uses:
model dir (runs/name), but not the model itself
dataset - bare, to index

Note: CAMs are stored as uint8, I convert them to float [0, 1] for plotting.
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
from PIL import Image
import cv2

if "PYCHARM_HOSTED" in os.environ:  # running in Pycharm
    matplotlib.use('module://backend_interagg')
elif "SCRATCHDIR" in os.environ:  # running on cluster
    pass
else:  # running in terminal
    matplotlib.use('tkagg')
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
import dataset_base
from dataset_base import show_labels_distribution, pick_dataset_version, load_annotations, label_names_binary, get_dataset_setup
from explanations import overlay_cam, sobel_edges, perturbation_masks, perturbation_baselines
from face_detection import get_ref_landmarks
from util_image import get_marker, plot_many, mix_mask, plot_many_df
from util_torch import init_seed, get_dataset_module

''' Global variables '''
run_dir = ''
nums_to_names = None
label_names = None
args = None
cam_dir = None
ext = 'pdf'  # extension for saving plots
percentages_kept = None  # deletion metric percentages [0, 100]
auc_percentages = None  # normalized to [0, 1]
cam_shape = None

parser = argparse.ArgumentParser()  # description='Process CAMs'
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
parser.add_argument('-z', '--show', help='show outputs', action='store_true')
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)
parser.add_argument('-n', '--no_log', help='do not save anything', action='store_true')
parser.add_argument('-p', '--path_prefix', help='path to dataset')
parser.add_argument('-f', '--files', help='path(s) to CAMs', type=str, nargs='+')


def read_cams(paths, path_prefix=None):
    """ Read pickled CAM DataFrame/s. """
    df = None
    if isinstance(paths, str):
        paths = [paths]

    if not isinstance(paths, list):
        raise ValueError(f'Invalid paths: {paths}')

    # concatenate dataframes
    lens = []
    for path in tqdm(paths):
        try:
            df_tmp = pd.read_pickle(path)
            source = basename(path)
            df_tmp['source'] = source  # broadcast to all rows

            print(f'Loading {source}, shape: {df_tmp.shape}')
            if df is None:
                df = df_tmp
                lens.append(len(df_tmp))
                continue
            ''' Check for overlapping lines by: idx, method '''
            intersection = df_tmp.merge(df, on=['idx', 'method'], how='inner')
            if not intersection.empty:
                print(f'Warning: {source} intersects with: {intersection["file_y"].unique()}')
            ''' Check for conflicting attributes - label, pred '''
            conflicting = (intersection['label_x'] != intersection['label_y']) | \
                          (intersection['pred_x'] != intersection['pred_y'])
            if any(conflicting):
                print(f'Warning: {source} and {intersection["file"].iloc[0]} have conflicting labels/preds')
            lens.append(len(df_tmp))

            df = pd.concat([df, df_tmp], ignore_index=True)
        except Exception as e:
            print(e)
            continue

    print(f'Merged {len(lens)}/{len(paths)} dataframes, total shape: {df.shape}, '
          f'original lengths: {lens} -> {sum(lens)}')

    ''' Change paths to images '''
    df['path'] = df['path'].apply(basename)
    if path_prefix is not None:
        df['path'] = df['path'].apply(lambda x: join(path_prefix, x))

    return df


# Calculations with CAMs
def cam_mean(df, select):
    """ Compute average CAM by class.
    :param df: dataframe with attr cam (n, c, h, w)
    :param select: 'label' or 'pred'
    :return: (c, h, w)
    """
    if select not in ['label', 'pred']:
        raise ValueError(f'cam_mean: invalid filtering classes selected: {select}')
    selected = np.stack(df[select].values)
    cams = np.stack(df.cam.values)
    cam = np.zeros(cam_shape)
    with warnings.catch_warnings():  # filter all RuntimeWarnings (mean of empty slice)
        warnings.filterwarnings(action='ignore')
        for i in range(len(label_names)):
            cam[i] = np.mean(cams[selected == i], axis=0)[i]
    cam[np.isnan(cam)] = 0
    # drop 'Other' class if empty
    if len(cam) == 5 and cam[4].sum() < 1e-2:
        cam = cam[:-1]
        print(f'cam mean dropped last class, now {cam.shape}')
    return cam


def cam_mean_confmat(df):
    cams, labels, preds = np.stack(df.cam.values), df.label.values, df.pred.values
    cam = np.zeros((cam_shape[0], *cam_shape))  # (pred, label, h, w)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        for i in range(len(label_names)):  # i == label (gt)
            for j in range(len(label_names)):  # j == pred
                cam[i, j] = np.mean(cams[(labels == i) & (preds == j)], axis=0)[j]

    cam[np.isnan(cam)] = 0
    if cam.shape[:2] == (5, 5) and cam[4].sum() < 1e-2 and cam[:, 4].sum() < 1e-2:
        cam = cam[:-1, :-1]
        print(f'cam mean confmat dropped last class, now {cam.shape}')
    return cam


def auc(probs, percentages=None):
    """ Compute the Area under Curve (AUC) for a given deletion/insertion metric curve.

    :param probs: (n) deletion/insertion scores for each percentage
    :param percentages: (n) percentages of image kept
    :return: AUC value
    """
    if percentages is None:
        percentages = auc_percentages
    # using negative percentages, because trapz() assumes increasing x-values
    return np.trapz(probs, -percentages)


# Analyses
def rank_by(df, group_by, key):
    return df.groupby(group_by)[key].mean().sort_values(ascending=False)


# Plotting building blocks
def plt_save_close(output_path=None):
    if output_path is not None and not args.no_log:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f'Saved to {output_path}')
    if args.show:
        plt.show()
    plt.close()


def plt_legend(order=None, line_labels=None, **legend_kwargs):
    plt.legend(**legend_kwargs)
    if order is not None:  # reorder legend by aucs (ascending)
        handles, labels_auto = plt.gca().get_legend_handles_labels()
        line_labels = line_labels if line_labels is not None else labels_auto  # use given labels if available
        plt.legend([handles[idx] for idx in order], [line_labels[idx] for idx in order], **legend_kwargs)


# Plotting full
def plot1xC(cams, title='', output_path=None):
    """Plot 5 CAMs in a row."""
    # todo very similar to `plot_cam_classes`
    fig, axs = plt.subplots(1, len(cams), figsize=(16, 4))  # , sharex=True)
    for jc, ax in zip(enumerate(cams), axs.flat):
        j, c = jc
        if c.shape[0] == 1:
            c = c[0]
        ax.imshow(c, vmin=0, vmax=1)
        ax.set_title(label_names[j])  # todo ensure matched binary x multiclass
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=-0.05)
    fig.suptitle(title)
    plt_save_close(output_path)


def plot_cam_classes(row, img, output_path=None, title='', **kwargs):
    """Plot image and CAMs (5-> 2x3, 2->1x3). """
    cam = row.cam
    if len(cam) == 5:
        nrows, ncols = 2, 3
        figsize = (12, 8)
    elif len(cam) == 4:
        nrows, ncols = 1, 5
        figsize = (12, 4)
    else:
        nrows, ncols = 1, 3
        figsize = (8, 3)
    plt.subplots(nrows, ncols, figsize=figsize)
    # pred_name = nums_to_names[row.pred]
    # gt_name = nums_to_names[row.label]
    method = row.method
    plt.suptitle(f'{method} idx:{row.idx}')
    ''' Image '''
    plt.subplot(nrows, ncols, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')
    ''' CAMs '''
    for j, c in enumerate(cam):
        if c.ndim == 3 and c.shape[0] == 1:
            c = c[0]  # remove channel dimension
        plt.subplot(nrows, ncols, j + 2)
        plt.imshow(c, **kwargs)  # uint8 heatmaps, don't normalize
        plt.axis('off')
        ax_title = nums_to_names[j]
        if j == row.label:
            ax_title += ' (GT)'
        if j == row.pred:
            ax_title += ' (pred)'
        plt.title(ax_title)

    plt.tight_layout()
    plt_save_close(output_path)


def plot_cam_mean_confmat(cams, title=None, output_path=None):
    """Plot confusion matrix of CAM means.

    cams: Input matrix: [c, c, h, w]
    """
    global label_names
    m, n = cams.shape[:2]
    fig, axs = plt.subplots(m, n, figsize=(12, 12))
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.995)

    for i in range(m):  # rows == gt
        class_gt = label_names[i]
        for j in range(n):  # cols == pred
            class_pred = label_names[j]
            if i == 0:  # top-row
                axs[i, j].set_title(class_pred, fontsize=12)
            if j == 0:  # left-column
                axs[i, j].set_ylabel(class_gt, rotation=90, labelpad=5, fontsize=12)
            c = cams[i, j]
            if c.shape[0] == 1:
                c = c[0]
            axs[i, j].imshow(c, vmin=0, vmax=1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # remove frame around image
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)

    # make plots tight but leave margin outside for the title, predicted, ground truth
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.93, left=0.07)

    # left of the plots: predicted
    axs[2, 0].text(-0.2, 0.98, 'predicted', rotation=90, va='center', ha='center', transform=axs[2, 0].transAxes,
                   fontsize=14)
    # above the plots: ground truth
    axs[0, 1].text(-0.1, 1.15, 'ground truth', va='center', ha='center', transform=axs[0, 2].transAxes, fontsize=14)
    plt_save_close(output_path)


def plot_baselines_variance(df, auc_type, output_path=None):
    """ Plot variance of perturbation scores for each baseline. """
    global percentages_kept
    attr_scores = 'del_scores_pred' if auc_type == 'auc_del' else 'ins_scores_pred'
    title_prefix = 'Deletion' if auc_type == 'auc_del' else 'Insertion'
    ''' Rank baselines based on variance of perturbation scores '''
    groups = df.groupby(['baseline'])[attr_scores]
    score_var = groups.apply(lambda x: np.var(np.stack(x.values), axis=0))
    score_means = score_var.apply(lambda x: np.mean(x)).sort_values()
    print(score_means)
    for i, m in zip(score_var.index, score_means):
        row_vals = score_var.loc[i]
        plt.plot(row_vals, label=f'{i}: {m:.3f}')
    plt_legend('baseline: mean variance', np.arange(len(score_means)))
    plt.title(f'Baseline {title_prefix} Metric Variance')
    plt.xticks(range(len(percentages_kept)), percentages_kept)
    plt.xlabel('% pixels kept')
    plt.ylabel(f'Variance')
    plt.tight_layout()
    plt_save_close(output_path)


def deletion_per_image(del_scores, line_labels=None, many_ok=False, title=None, output_path=None):
    """ Plot deletion metric for each image.
        Parameters:
    del_scores (list): deletion scores.
    line_labels (Series, optional): labels for each score line.
    many_ok (bool, optional): allow plotting a large number of samples.
    title (str, optional): Title
    output_path (str, optional): Path to save the output plot.
    """
    global percentages_kept
    if len(del_scores) > 20 and not many_ok:
        print(f'deletion_per_image: too many {len(del_scores)} samples, '
              f'override by `many_ok` to run anyway.')
        return
    if isinstance(line_labels, pd.Series):
        line_labels = line_labels.values
    plt.figure(figsize=(6, 4))
    for i, scores_line in enumerate(del_scores):
        kwargs = {}
        if line_labels is not None and len(line_labels) > i:
            kwargs = {'label': line_labels[i], 'marker': get_marker(line_labels[i])}
        plt.plot(percentages_kept, scores_line, **kwargs)
    plt.xticks(percentages_kept)  # original x-values ticks
    plt.gca().invert_xaxis()  # decreasing x axis
    plt.ylim(0, 1.05)
    if len(del_scores) <= 10 and line_labels is not None:  # otherwise too many labels
        plt.legend()
    plt.ylabel('Prediction Score')
    plt.xlabel('Perturbation Intensity (% of image kept)')
    plt.title(f'Deletion Metric {title}')
    # remove top and right spines
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt_save_close(output_path)


def perturbation_curve_mean(df, key, auc_type, title_suffix='', line_labels=None, output_path=None, ranges=False, sort_aucs=True, legend_out=False):
    """ Plot deletion/insertion metric by key, averaged over samples. """
    if auc_type not in ['auc_del', 'auc_ins']:
        raise ValueError(f'Invalid del_ins: {auc_type}')
    attr_scores = f'{auc_type[len("auc_"):]}_scores_pred'
    title_prefix = 'Deletion' if auc_type == 'auc_del' else 'Insertion'
    key_unique = np.sort(df[key].unique())
    key_capitalised = key[0].upper() + key[1:]
    aucs = []
    # place legend out to the right if requested
    legend_kwargs = {'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'} if legend_out else {}
    legend_kwargs = {'title': f'{key_capitalised}: AUC', **legend_kwargs}
    plt.figure(figsize=(10 if legend_out else 7, 5))
    for i, k_val in enumerate(key_unique):
        line_label = line_labels[i] if line_labels is not None else k_val
        df_filtered = df[df[key] == k_val]
        scores = np.stack(df_filtered[attr_scores].values)
        ys = np.mean(scores, axis=0)
        auc_mean = df_filtered[auc_type].mean()
        aucs.append(auc_mean)
        plt.plot(percentages_kept, ys, label=f'{line_label}: {auc_mean:.3f}')
        if ranges:
            stds = np.std(scores, axis=0)
            plt.fill_between(percentages_kept, ys - stds, ys + stds, alpha=0.2)
    plt.title(f'{title_prefix} Metric by {key_capitalised}' + title_suffix)
    plt.xlabel('% pixels kept')
    plt.ylabel('Prediction Score')
    plt.xticks(percentages_kept)  # original x-values ticks
    order = np.argsort(aucs)
    if auc_type == 'auc_del':
        plt.gca().invert_xaxis()  # decreasing x-axis for deletion
    else:  # decreasing order for insertion
        order = order[::-1]
    plt.ylim(0, 1.01)
    plt_legend(order if sort_aucs else None, None, **legend_kwargs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt_save_close(output_path)


# unused: replaced by perturbation_curve_mean, which is more general
def perturbation_curve_mean_per_pred(df, auc_type, title=None, output_path=None, ranges=False, sort_aucs=True):
    """ Plot Deletion metric by predicted class, """
    if auc_type not in ['auc_del', 'auc_ins']:
        raise ValueError(f'Invalid del_ins: {auc_type}')
    title_prefix = 'Deletion' if auc_type == 'del' else 'Insertion'
    attr_scores = f'{auc_type[len("auc_"):]}_scores_pred'
    scores = df[attr_scores].values
    preds = df.pred.values
    aucs = []
    for i, label in enumerate(label_names):
        scores_per_class = scores[preds == i]
        if len(scores_per_class) == 0:
            continue
        scores_per_class = np.stack(scores_per_class)
        mean_y = np.mean(scores_per_class, axis=0)
        auc_mean = auc(mean_y)
        aucs.append(auc_mean)
        plt.plot(percentages_kept, mean_y, label=f'{label}: {auc_mean:.3f}')
        if ranges:  # plot std
            std_y = np.std(scores_per_class, axis=0)
            plt.fill_between(percentages_kept, mean_y - std_y, mean_y + std_y, alpha=0.2)
    plt.xticks(percentages_kept)  # original x-values ticks
    plt.ylim(0, 1.01)
    order = np.argsort(aucs)
    if auc_type == 'auc_del':
        plt.gca().invert_xaxis()
    else:  # decreasing order for insertion
        order = order[::-1]
    plt_legend(f'predicted: AUC', order if sort_aucs else None)
    plt.xlabel('% pixels kept')
    plt.ylabel('Prediction Score')
    plt.title(f'{title_prefix} Metric by Prediction' + title)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt_save_close(output_path)


# def main():
#     global auc_percentages, percentages_kept  # only uncomment when using def main()
if __name__ == '__main__':
    """
    Process CAMs - perform analyses, visualize, and save plots.
    """
    ''' Arguments '''
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))

    ''' Process paths and filenames '''
    # paths like: 'runs/sage-cherry-91/eval-rose_youtu-single-all_attacks-all/cam/cams_SobelFakeCAM_20.pkl.gz'
    if "PYCHARM_HOSTED" in os.environ:
        args.files = glob(args.files[0])  # glob expand file argument

    ''' Directories/paths '''
    cam_dir = dirname(args.files[0])
    path_list = args.files[0].split('/')  # assume common directory prefix for cam files
    run_dir = join(*path_list[:2])  # rebuild runs/<run_name>
    eval_dir = path_list[2]
    # eval directory name format to parse, hyphen-separated:
    # 'eval', dataset_name, dataset_note [, attack], mode
    _, dataset_name, dataset_note, mode, attack = eval_dir.split('-')
    # output directory for sample images
    samples_dir = join(cam_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    ''' Parse filenames '''
    # filenames example: cams_EigenCAM_20.pkl.gz
    # filenames pattern: cams_<method>_<limit>.pkl.gz
    filenames = [basename(f) for f in args.files]
    # read all limit strings from filenames
    limits = set(f.split('_')[2].split('.')[0] for f in filenames)
    if len(limits) != 1:
        raise ValueError(f'Multiple different limits in filenames: {limits}')

    methods_filenames = [f.split('_')[1] for f in filenames]
    limit = limits.pop()  # only one in the set (which can also be destroyed)
    limit = -1 if limit == 'all' else int(limit)

    ''' Load CAMs '''
    df = read_cams(args.files, args.path_prefix)
    df_untouched = df.copy()
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    print('Loading model and setup from:', run_dir)

    ''' Initialization '''
    seed = args.seed if args.seed is not None else config.seed_eval_default  # 42
    init_seed(seed)
    ''' Plotting params '''
    sns.set_style('whitegrid')
    # First plots are charts, keep grid on. (Then turn it off for images)
    plt.rcParams['axes.grid'] = True
    sns.set_context("paper", font_scale=1.5)

    ''' Load Dataset '''
    if True:
        # training_mode = config_dict['mode']  # 'all_attacks'
        # dataset_name = config_dict['dataset']  # 'rose_youtu'
        dataset_module = get_dataset_module(dataset_name)
        dataset_meta = pick_dataset_version(dataset_name, mode, attack, dataset_note)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']
        split = 'test'  # implicitly always working with test split (unseen data)
        annotations = load_annotations(dataset_meta, seed, args.path_prefix, limit)[split]
        ds = dataset_module.Dataset(annotations)
        bona_fide = dataset_module.bona_fide
        label_names, num_classes = get_dataset_setup(dataset_module, mode)
        nums_to_names = dataset_module.nums_to_unified if mode == 'all_attacks' else dataset_base.nums_to_names_binary
        nums_to_unified = dataset_module.nums_to_unified
        attack_test_name = nums_to_unified[attack_test] if attack_test != 'all' else 'all'
        ''' Show labels distribution '''
        show_labels_distribution(annotations['label'], split, num_classes)

    ''' Metadata '''
    # join CAMs with metadata from annonations file
    df = df.merge(annotations, on='idx', how='inner', suffixes=('', '_y'))

    # Check CAMs shapes
    df['shp'] = df.cam.apply(lambda x: x.shape)
    if len(df.shp.unique()) != 1:
        raise ValueError(f'Different CAM shapes!\n{df.shp.value_counts()}')
    cam_shape = df.iloc[0]['shp']
    df_shapes_by_method = df[['method', 'shp']].groupby('shp')
    # which shapes are there for which methods?
    for g, rows in df_shapes_by_method:
        print(f'{g}:\n', rows.method.value_counts())

    if len(df_shapes_by_method) != 1:
        filter_idxs = df.shp == cam_shape
        print(f'Filtering CAMs to keep only the first shape {sum(filter_idxs)}/{len(filter_idxs)}.')
        df = df[df.shp == cam_shape]
    print(f'CAM shape: {cam_shape}')

    ''' AUC for each CAM '''
    percentages_kept = df['percentages_kept'].values[0]  # assume same for all
    auc_percentages = np.array(percentages_kept) / 100  # normalize to [0, 1]
    # deletion scores for predicted class
    df['del_scores_pred'] = df.apply(lambda row: row.del_scores[:, row.pred], axis=1)
    df['ins_scores_pred'] = df.apply(lambda row: row.ins_scores[:, row.pred], axis=1)
    df['auc_del'] = df['del_scores_pred'].apply(auc)
    df['auc_ins'] = df['ins_scores_pred'].apply(auc)
    ''' Expand annotations '''
    df['correct'] = df['label'] == df['pred']
    if cam_shape[0] == 5:
        # drop 5th class 'Other', unused.
        df.cam = df.cam.apply(lambda x: x[:-1])
    df.cam = df.cam.apply(lambda x: np.float32(x) / 255)  # from uint8 to [0, 1] float
    df['cam_pred'] = df.apply(lambda row: row.cam[row.pred], axis=1)  # (H, W)
    cams = np.stack(df['cam'].values)
    labels = df['label'].values
    preds = df['pred'].values
    idxs = df['idx'].values

    methods_unique = np.sort(df['method'].unique())
    baselines_unique = np.sort(df['baseline'].unique())
    sample_row = df.iloc[0]
    sample_img = Image.open(sample_row.path)
    img_shape = sample_img.size  # (H, W), assume same for all
    landmarks = get_ref_landmarks()  # (5, 2) = [[x, y], ...] = [w, h] <- the only exception in axis order!
    ###############################################################################
    print(f'Loaded {len(df)} CAMs from {len(args.files)} files.')
    print(f'Methods: {methods_unique}')
    # temporary: rename baselines:
    # drop rows with baseline 'blurdarker'
    df = df.drop(df[df['baseline'] == 'blurdarker'].index)
    mapping = {'blurdark': 'blur_div4', 'blurdarkerer': 'blur_div8'}
    df.baseline = df.baseline.apply(lambda x: mapping[x] if x in mapping else x)
    print(f'Baselines: {baselines_unique}')
    print(f'Dataset, variant, split: {dataset_name}, {dataset_note}, {split}')
    print(f'Limit: {limit}')
    print(f'Training Mode, test attack: {mode}, {attack_test_name}')
    print(f'Ground-truth labels: {df.label.unique()}')
    print(f'Showing plots: {args.show}')

    ''' ------- Exploring CAMs ------- '''
    dfcopy = df.copy()
    df_sample = df.sample(10)

    ''' Rank by AUC '''
    for auc_type in ['auc_del', 'auc_ins']:
        auc_methods_ranking = rank_by(df, ['method'], auc_type)
        print(auc_methods_ranking, '\n')
        if len(baselines_unique) > 1:
            auc_baselines_ranking = rank_by(df, ['baseline'], auc_type)
            auc_methods_baselines_ranking = rank_by(df, ['method', 'baseline'], auc_type)
            print(auc_baselines_ranking, '\n\n', auc_methods_baselines_ranking, '\n')

    ''' Does image blurriness matter? '''
    laplacian_per_label_correct = df.groupby(['correct', 'label']).laplacian_var.mean()
    # not so different, and not enough samples to support the hypothesis => no.

    ''' Does perturbation AUC correlate with accuracy? '''
    acc_by_auc = df.groupby(['correct'])[['auc_del', 'auc_ins']].mean()
    print(acc_by_auc)
    # Yes! The higher the AUC, the higher the accuracy.
    # It makes sense, because the model is more confident in its prediction,
    # even when the image is perturbed.

    ''' Plot baseline deletion scores variances (how consistent is the baseline) '''
    if len(baselines_unique) > 1:
        for auc_type in ['auc_del', 'auc_ins']:
            output_path = join(cam_dir, f'baseline_{auc_type}_scores_var.{ext}')
            plot_baselines_variance(df, auc_type, output_path)
            # black has the lowest variance, which is good.

    ''' Plot CAMs as used in the deletion metric. '''
    if False:
        row = df[df.label_orig == 7].iloc[41]
        idx = row.idx
        method = row.method
        cam_pred = row.cam_pred
        cam_pred_copy = cam_pred.copy()
        img_pil = Image.open(join(args.path_prefix, row['path']))
        img = np.array(img_pil, dtype=float) / 255  # HWC
        img_chw = np.transpose(img, (2, 0, 1))  # CHW
        ''' Deletion metric perturbations '''
        cam_pred = cv2.resize(cam_pred, (img.shape[0], img.shape[1]))  # interpolation bilinear by default
        masks = perturbation_masks(cam_pred, percentages_kept)
        baselines = perturbation_baselines(img_chw)
        baseline = baselines[row.baseline]
        imgs_perturbed = [(img_chw * mask + (1 - mask) * baseline) for mask in masks]
        ''' Compare baselines '''
        plot_many([img, *list(baselines.values())], title='Perturbation baselines',
                  titles=['image', *list(baselines.keys())],
                  output_path=join(cam_dir, f'baselines_idx{idx}.{ext}'))
        # CAM - low and upsampled resolution
        plot_many([img, cam_pred_copy, cam_pred], title=f'{row.idx} {row.method}',
                  titles=['image', 'CAM', 'CAM upsampled to image'],
                  output_path=join(cam_dir, f'cam_upsampled.{ext}'))
        # perturbed images
        pct_strings = [f'{p}%' for p in percentages_kept]
        plot_many(imgs_perturbed[:-1:2], titles=pct_strings[:-1:2], title='perturbed images', rows=1,
                  output_path=join(cam_dir, f'perturbed_images_idx{idx}_{method}.{ext}'))
        plot_many(masks[:-1:2], titles=pct_strings[:-1:2], title='perturbation masks', rows=1,
                  output_path=join(cam_dir, f'perturbation_masks_idx{idx}_{method}.{ext}'))

        # compare Sobel edges: original resolution->downsampled x low-resolution
        cam_hw = cam_pred_copy.shape
        sobel_full = sobel_edges(img)
        sobel_downsampled = cv2.resize(sobel_full, cam_hw, interpolation=cv2.INTER_AREA)
        img_lr = cv2.resize(img, cam_hw)
        sobel_lr = sobel_edges(img_lr)
        plot_many([img, sobel_full, sobel_downsampled, img_lr, sobel_lr], rows=1,
                  titles=['image', 'sobel', 'sobel downsampled', 'image downsampled', 'sobel low-res'],
                  output_path=join(cam_dir, f'sobel_edges_downscale.{ext}'))

    ''' Find explanations that cause dropoff >= 0.5 when perturbed '''
    visible, muted = 1, 0.3
    pct_kept = np.array(percentages_kept)
    ''' Plot dropoff explanations '''
    if False:
        ''' 
        How much does it take to change the prediction?
        find index in del_scores, where the predicted class probability drops below 0.5 *
        get corresponding percentage of pixels kept
        get perturbation mask for the CAM (of the originally predicted class) at that percentage
        apply the mask to the image, showing the explanation, with the rest of the image muted.
        
        * 0.5 value -> almost sure prediction changed (binary = sure). 
        I could also look for first changed class in del_scores.
        
        Next, I could group the rows by sample path, and show them together, comparing the methods.
        '''
        df['dropoff_idx'] = df.apply(lambda row: np.argmax(row.del_scores_pred < 0.5), axis=1)  # argmax -> first
        df['dropoff_pct'] = pct_kept[df.dropoff_idx]
        # if prediction never drops below 0.5, then argmax returns 0 => dropoff_cpt == 100%. Set to 0%.
        df[df.dropoff_pct == 100] = 0
        if False:
            pass
            # descs = []
            # for _, row in df_kept.iterrows():
            #     pred_name = nums_to_names[row.pred]
            #     pred_at_dropoff = row.del_scores[row.dropoff_idx]
            #     pred_name_dropoff = nums_to_names[np.argmax(pred_at_dropoff)]
            #     desc = f'{row.idx}: kept {row.dropoff_pct}% {pred_name} -> {pred_name_dropoff}'
            #     descs.append(desc)

        dropoffs_dir = join(cam_dir, 'dropoff')
        os.makedirs(dropoffs_dir, exist_ok=True)
        ''' Plot dropoff explanations '''
        b = df.iloc[0].baseline  # assuming same baseline for all rows
        print(f'Dropoff explanations: using baseline "{b}", out of {baselines_unique}')
        paths_kept = df[df.dropoff_pct >= 80].path.unique()
        paths_kept_correct = df[(df.dropoff_pct >= 80) & (df.correct)].path.unique()
        paths_kept_maskcropped = df[(df.dropoff_pct >= 60) & (df.label_orig == 5)].path.unique()
        paths_kept_masktop = df[(df.dropoff_pct >= 60) & (df.label_orig == 7)].path.unique()
        paths_kept_incorrect = df[(df.dropoff_pct >= 80) & (df.correct == False)].path.unique()
        for path in paths_kept_masktop:
            df_path = df[(df.path == path) & (df.baseline == b)]
            row = df_path.iloc[0]
            idx = row.idx
            img = np.array(Image.open(path), dtype=float) / 255
            # get dropoff explanation masks
            expls = df_path.apply(lambda row: perturbation_masks(cv2.resize(row.cam_pred, img_shape), [row.dropoff_pct])[0][..., None], axis=1)
            # blend masks with image
            blends = [mix_mask(img, (1 - expl)) for expl in expls]
            gt_name, pred_orig_name = nums_to_unified[row.label_unif], nums_to_names[row.pred]
            title = f'GT: {gt_name} pred: {pred_orig_name}'  # ground truth always as multi-class for clarity
            pred_new_names = [nums_to_names[np.argmax(row.del_scores[row.dropoff_idx])] for _, row in df_path.iterrows()]
            # title = []
            # for _, row in df_path.iterrows():
            #     pred_new_name = nums_to_names[np.argmax(row.del_scores[row.dropoff_idx])]
            #     title.append(f'{row.method} {100 - row.dropoff_pct:d}% {pred_new_name}')
            # titles = [f'image idx={idx}'] + [f'{row.method} {100 - row.dropoff_pct:d}%}' for _, row in df_path.iterrows()]
            titles = [f'{pred_orig_name}\noriginal'] + [f'{nums_to_names[np.argmax(row.del_scores[row.dropoff_idx])]}\n{row.method}' for _, row in
                                                        df_path.iterrows()]
            plot_many([img, *blends], titles=titles, title=title, output_path=join(dropoffs_dir, f'del_idx{idx}_gt{gt_name}_pred{pred_name}.{ext}'))
            s = input()

        titles = [f'{100 - row.dropoff_pct:d}%' for _, row in df_kept.iterrows()]
        imgs = [Image.open(row.path) for _, row in df_kept.iterrows()]
        imgs = [np.array(img, dtype=float) / 255 for img in imgs]
        imgs_masked = [(im * mask * visible + (1 - mask) * im * muted) for im, mask in zip(imgs, dropoff_masks)]
        plot_many(imgs_masked, titles=titles, title='Dropoff masks')
        mask_removed = [(im * mask * muted + (1 - mask) * 1 * im * visible) for im, mask in zip(imgs, dropoff_masks)]
        plot_many(mask_removed, titles=descs, title='Dropoff explanations',
                  output_path=join(cam_dir, f'dropoff_explanations.{ext}'))

    ''' Low face probability by MTCNN'''
    if False:
        df_lowprob = df.copy().groupby('path').agg({'face_prob': 'min'}) \
                         .reset_index().sort_values('face_prob')[:10]

        df.face_prob = df.face_prob.astype(float)
        df_lowprob = df.copy()
        df_grouped = df_lowprob.groupby('path')['face_prob'].idxmin()  # get the indices of min 'face_prob'
        df_lowprob = df_lowprob.loc[df_grouped].sort_values('face_prob')[:10]
        plot_many_df(df_lowprob, titles=[f'{f:.2f}' for f in df_lowprob.face_prob], title='Low MTCNN face probability', rows=2,
                     output_path=join(cam_dir, f'low_face_prob.{ext}'))

    for _, row in df_lowprob.iterrows():
        img = np.array(Image.open(row.path)).astype(float) / 255
        plot_many([img, *row.cam])
        _ = input()

    # how does the face_prob correlate with the deletion / insertion metric?

''' Landmarks '''
if False:
    # sample image with reference landmarks
    img = Image.open(df.iloc[0].path)
    plt.imshow(img)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=1, c='r')
    plt.show()

''' Best and worst CAMs per method, ranked by AUC '''
if False:
    key = 'method'
    key_unique = df[key].unique()
    n_samples = 5
    for k_val in key_unique:
        df_k = df[df[key] == k_val].copy()
        df_k = df_k.sort_values('auc_del')  # ascending
        worst = df_k.head(n_samples)
        best = df_k.tail(n_samples)
        wb = pd.concat([worst, best], ignore_index=True)
        imgs = [Image.open(row['path']) for _, row in wb.iterrows()]
        imgs = [np.array(img, dtype=float) / 255 for img in imgs]

        # keep only area highlighted by cam_pred
        cam_upsampled = wb.cam_pred.apply(lambda x: cv2.resize(x, img_shape))
        cam_upsampled = np.stack(cam_upsampled, dtype=float)[..., None]  # (n, h, w, 1)
        imgs_masked_by_cam = [im * cam for im, cam in zip(imgs, cam_upsampled)]

        # plot_many(imgs, titles=[f'{line_auc:.3f}' for line_auc in wb.auc_del])
        plot_many(imgs_masked_by_cam, titles=[f'{a:.3f}' for a in wb.auc_del], title=f'{key}: {k_val}')
        # deletion_per_image(wb.del_scores, labels=[f'{r.idx}: {r.auc_del:.3f}' for _, r in wb.iterrows()], title=f'- {m}')

if False:
    # best CAMs per class (within method)
    for m in methods_unique:
        pass

''' Plot CAMs for wrong predictions '''
if False:
    pass

''' Across-method correlation '''
if False:
    from scipy.stats import pearsonr, spearmanr

    # step 1: single sample
    idx = df.iloc[0].idx
    ds_idx = [sample['idx'] for sample in ds]
    sample = ds[ds_idx.index(idx)]
    img = sample['image']

    # step 2: get all cams for this sample
    df_sample = df[df.idx == idx]
    methods = df_sample.method.values
    cams_sample = np.stack(df_sample.cam.values)
    cams_sample = cams_sample[:, :, 0, ...]  # drop 1-channel dim  -- not applicable
    cam1 = cams_sample[0]
    cam2 = cams_sample[1]
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
            # verify everything's normalized to [0, 1]
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
    df10 = df[:10]
    output_path = join(cam_dir, f'deletion_metric_samples.{ext}')
    deletion_per_image(df10.del_scores, line_labels=df10.idx, output_path=output_path)

''' Perturbation Metric Plot (deletion and insertion) - compare methods '''
if True:
    for auc_type in ['auc_del', 'auc_ins']:
        output_path = join(cam_dir, f'{auc_type}_metric_methods.{ext}')
        perturbation_curve_mean(df, 'method', auc_type, '', legend_out=True, ranges=False, output_path=output_path)

''' Deletion Metric Plot - per-class, for each baseline '''
if True:
    """ 
    1 plot per baseline, 1 line per class.  
    Compare ways of perturbing the image with different baselines.
    """
    for baseline in baselines_unique:
        for auc_type in ['auc_del', 'auc_ins']:
            df_base = df[df.baseline == baseline]
            output_path = join(cam_dir, f'{auc_type}_{baseline}_classes.{ext}')
            # perturbation_curve_mean_per_pred(df_base, auc_type, f' - baseline: {baseline}', output_path, ranges=False, sort_aucs=True)
            perturbation_curve_mean(df_base, 'pred', auc_type, f' - baseline: {baseline}', label_names, output_path, ranges=False, sort_aucs=True)

# Plotting images, turn off grid
plt.rcParams['axes.grid'] = True
# set default matplotlib imshow colormap
plt.rcParams['image.cmap'] = 'viridis'

''' Per-image CAM - per-class heatmaps '''
kwargs_cams = {'vmin': 0, 'vmax': 1, 'cmap': 'viridis'}
if False:
    b = df.iloc[0].baseline  # assuming same baseline for all rows
    print(f'Showing per-image CAMs for baseline "{b}" only')
    df3 = df[df.baseline == b].sample(3)
    for i, row in df3.iterrows():
        img = np.array(Image.open(row.path), dtype=float) / 255
        output_path = join(cam_dir, f'{row.method}_idx{row.idx}.{ext}')
        plot_cam_classes(row, img, output_path, **kwargs_cams)

''' Average CAM by predicted category '''
if False:
    cams_pred = cam_mean(df, 'pred')
    output_path = join(cam_dir, f'avg_class_pred.{ext}')
    plot1xC(cams_pred, 'Average CAM per predicted class', output_path)
    plot_many([*cams_pred], title='Average CAM per predicted class', titles=label_names, output_path=output_path, rows=1)

''' Average CAM by ground-truth category '''
if False:
    cams_gt = cam_mean(df, 'label')
    output_path = join(cam_dir, f'avg_class_label.{ext}')
    plot1xC(cams_gt, 'Average CAM per ground_truth class', output_path)

''' Average CAM per (predicted, ground-truth) category '''
if False:
    # confusion matrix for CAMs
    cams_confmat = cam_mean_confmat(df)
    output_path = join(cam_dir, f'avg_confmat.{ext}')
    plot_cam_mean_confmat(cams_confmat, 'Average CAM Confusion Matrix', output_path)

''' Incorrect predictions: image, predicted and ground truth CAMs '''
if False:
    limit_incorrect = 20
    df_filtered = df[df.correct == False]
    print(f'Incorrect predictions: {len(df_filtered)} / {len(df)}')
    if len(df_filtered) > limit_incorrect:
        print(f'Too many incorrect predictions to plot, limiting to {limit_incorrect}')
        df_filtered = df_filtered.sample(limit_incorrect)
    for i, row in df_filtered.iterrows():
        idx = row.idx
        img = Image.open(row.path)
        pred = row.pred
        label = row.label
        output_path = join(samples_dir, f'incorrect_idx{idx}_pred{pred}_gt{label}_{row.method}.{ext}')
        ''' Overlay CAM '''
        overlayed_pred = overlay_cam(img, row.cam[pred])
        overlayed_label = overlay_cam(img, row.cam[label])
        fig_title = f'Prediction Error'
        # 1x3 figure: image, predicted cam, ground truth cam
        imgs = [img, overlayed_pred, overlayed_label]
        titles = [f'Image[{idx}]', f'Predicted: {nums_to_names[pred]}', f'Ground truth: {nums_to_names[label]}']
        if False:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            for ax, img, title in zip(axs, imgs, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')

            plt.suptitle(fig_title)
            plt.tight_layout()
            plt_save_close(output_path)
        plot_many(imgs, titles=titles, output_path=output_path)

''' Further possible comparisons '''
''' plot deletion scores per-baseline (old) '''
if False:
    cams_df = df
    baselines = cams_df.baseline.unique()
    xs = cams_df.percentages_kept.iloc[0]

    fig, ax = plt.subplots(len(baselines), 1, figsize=(6, 10), sharex=True)
    for n_samples, base_name in enumerate(baselines):
        ax[n_samples].set_title(base_name)
        cams_df_base = cams_df[cams_df.baseline == base_name]
        del_scores = np.stack(cams_df_base.del_scores.to_numpy())
        for i, c in enumerate(label_names):
            idxs = cams_df_base.label == i
            dsi = del_scores[idxs]
            ax[n_samples].plot(xs, dsi.mean(axis=0), label=c)

    plt.ylabel('Score')
    plt.xlabel('% Pixels Kept')
    plt.suptitle('Deletion Scores per Baseline')
    plt.gca().invert_xaxis()  # reverse x-axis
    # add minor y ticks

    # figure legend out right top
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()
