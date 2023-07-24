"""
cams.pkl.gz:
['cam', 'idx', 'path', 'method', 'percentages_kept', 'baseline', 'label',
   'pred', 'del_scores', 'pred_scores', 'source']
joined with annotation paths:
['label_text', 'speaking', 'device', 'glasses', 'environment', 'id1', 'id2',
   'path_y', 'box_orig', 'box', 'landmarks', 'dim_orig', 'face_prob',
   'laplacian_var', 'label_orig', 'label_unif', 'label_bin', 'label_y',]
computed:
['shp', 'del_scores_pred', 'auc', 'cam_pred']

I need:
model dir (runs/name), but not the model itself
dataset - bare, to index

Note: CAMs are stored as uint8, but I convert them to float32 [0, 1] for plotting.
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
from util_image import get_marker, plot_many
from util_torch import init_seed, get_dataset_module


''' Global variables '''
run_dir = ''
nums_to_names = None
label_names = None
args = None
cam_dir = None
ext = 'png'  # extension for saving plots
percentages_kept = None  # deletion metric percentages [0, 100]
auc_percentages = None  # normalized to [0, 1]

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
def cam_mean(cams, select):
    """ Compute average CAM by class.
    :param cams: (n, c, h, w)
    :param select: (n) predicted or ground-truth classes
    :return: (c, h, w)
    """
    cam = np.zeros(cam_shape)
    with warnings.catch_warnings():  # filter all RuntimeWarnings (mean of empty slice)
        warnings.filterwarnings(action='ignore')
        for i in range(len(label_names)):
            cam[i] = np.mean(cams[select == i], axis=0)[i]
    cam[np.isnan(cam)] = 0
    return cam


def cam_mean_confmat(cams, labels, preds):
    cam = np.zeros((len(label_names), *cam_shape))  # (pred, label, h, w)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        for i in range(len(label_names)):  # i == label (gt)
            for j in range(len(label_names)):  # j == pred
                cam[i, j] = np.mean(cams[(labels == i) & (preds == j)], axis=0)[j]
    cam[np.isnan(cam)] = 0
    return cam


def auc(deletion_scores, percentages=None):
    """ Compute the Area under Curve (AUC) for a given deletion curve. """
    if percentages is None:
        percentages = auc_percentages
    # using negative percentages, because trapz() assumes increasing x-values
    return np.trapz(deletion_scores, -percentages)


# Parameterised analyses
def rank_by_auc(df, columns):
    return df.groupby(columns)['auc'].mean().sort_values(ascending=False)


# Plotting building blocks
def plt_save_close(output_path=None):
    if output_path is not None and not args.no_log:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f'Saved to {output_path}')
    if args.show:
        plt.show()
    plt.close()


def plt_legend(legend_title, order=None, line_labels=None):
    plt.legend(title=legend_title)
    if order is not None:  # reorder legend by aucs (ascending)
        handles, labels_auto = plt.gca().get_legend_handles_labels()
        line_labels = line_labels if line_labels else labels_auto  # use given labels if available
        plt.legend([handles[idx] for idx in order], [line_labels[idx] for idx in order], title=legend_title)


# Plotting full
def plot1x5(cams, title='', output_path=None):
    """Plot 5 CAMs in a row."""
    fig, axs = plt.subplots(1, 5, figsize=(16, 4))  # , sharex=True)
    for jc, ax in zip(enumerate(cams), axs.flat):
        j, c = jc
        if c.shape[0] == 1:
            c = c[0]
        ax.imshow(c, vmin=0, vmax=1)
        ax.set_title(label_names[j])
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=-0.05)
    fig.suptitle(title)
    plt_save_close(output_path)


def plot2x3(row, img, output_path=None, title='', **kwargs):
    """Plot image and CAMs (5-> 2x3, 2->1x3). """
    cam = row.cam
    if len(cam) == 5:
        nrows, ncols = 2, 3
        figsize = (12, 8)
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


def plot5x5(cams, title=None, output_path=None):
    """Plot 5x5 figure with a confusion matrix of CAMs.

    cams: Input matrix: [gt, pred, h, w]
    """
    global label_names
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    if title is not None:
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
            axs[i, j].imshow(c, vmin=0, vmax=1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # remove frame around image
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    # write at the left: "predicted"
    axs[2, 0].text(-0.2, 0.5, 'predicted', rotation=90, va='center', ha='center', transform=axs[2, 0].transAxes,
                   fontsize=14)
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


def deletion_metric_per(df, key, output_path=None, line_labels=None, ranges=True, sort_aucs=True):
    key_unique = df[key].unique()
    aucs = []
    for k_val in key_unique:
        df_filtered = df[df[key] == k_val]
        del_scores = np.stack(df_filtered.del_scores_pred.values)
        ys = np.mean(del_scores, axis=0)
        auc_mean = df_filtered.auc.mean()
        aucs.append(auc_mean)
        plt.plot(percentages_kept, ys, label=f'{k_val}: {auc_mean:.3f}')
        if ranges:
            stds = np.std(del_scores, axis=0)
            plt.fill_between(percentages_kept, ys - stds, ys + stds, alpha=0.2)
    plt.title(f'Deletion metric per {key}')
    plt.xlabel('% pixels kept')
    plt.ylabel('Prediction Score')
    plt.xticks(percentages_kept)  # original x-values ticks
    plt.gca().invert_xaxis()  # decreasing x axis
    plt.ylim(0, 1.05)
    plt_legend(f'{key}: AUC', np.argsort(aucs) if sort_aucs else None, line_labels)
    plt.tight_layout()
    plt_save_close(output_path)


def deletion_metric_per_pred(df, output_path=None, title=None, ranges=True, sort_aucs=True):
    """ Plot Deletion metric by predicted class, """
    del_scores = df['del_scores_pred'].values
    preds = df['pred'].values
    aucs = []
    for i, label in enumerate(label_names):
        y_pred_class = del_scores[preds == i]
        if len(y_pred_class) == 0:
            continue
        y_pred_class = np.stack(y_pred_class)
        mean_y = np.mean(y_pred_class, axis=0)
        auc_mean = auc(mean_y)
        aucs.append(auc_mean)
        plt.plot(percentages_kept, mean_y, label=f'{label}: {auc_mean:.3f}')
        if ranges:  # plot std
            std_y = np.std(y_pred_class, axis=0)
            plt.fill_between(percentages_kept, mean_y - std_y, mean_y + std_y, alpha=0.2)
    plt.xticks(percentages_kept)  # original x-values ticks
    plt.gca().invert_xaxis()  # decreasing x axis
    plt.ylim(0, 1.05)
    plt_legend(f'predicted: AUC', np.argsort(aucs) if sort_aucs else None)
    plt.xlabel('% pixels kept')
    plt.ylabel('Prediction Score')
    plt.title(f'Deletion Metric by Class' + title)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt_save_close(output_path)


# def main():
if __name__ == '__main__':
    # global auc_percentages, percentages_kept  # only uncomment when using def main()
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
    sns.set_style('whitegrid')  # plotting params
    plt.rcParams['axes.grid'] = True  # gridlines initially on

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
        label_names, num_classes = get_dataset_setup(dataset_name, mode)
        nums_to_names = dataset_module.nums_to_unified if mode == 'all_attacks' else dataset_base.nums_to_names_binary
        nums_to_unified = dataset_module.nums_to_unified

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
    df['auc'] = df['del_scores_pred'].apply(auc)
    ''' Expand annotations '''
    df['correct'] = df['label'] == df['pred']
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
    df['baseline'] = df['baseline'].apply(lambda x: mapping[x] if x in mapping else x)
    print(f'Baselines: {baselines_unique}')
    print(f'Dataset, variant, split: {dataset_name}, {dataset_note}, {split}')
    print(f'Limit: {limit}')
    print(f'Training Mode, test attack: {mode}, {nums_to_unified[attack_test]}')
    print(f'Ground-truth labels: {df.label.unique()}')
    print(f'Showing plots: {args.show}')

    ''' ------- Exploring CAMs ------- '''
    # First plot charts, keep grid on. (Then turn it off for images)
    plt.rcParams['axes.grid'] = True

    dfcopy = df.copy()
    df_sample = df.sample(10)

    ''' Rank by AUC '''
    auc_methods_ranking = rank_by_auc(df, ['method'])
    auc_baselines_ranking = rank_by_auc(df, ['baseline'])
    auc_methods_baselines_ranking = rank_by_auc(df, ['method', 'baseline'])
    print(auc_methods_ranking, '\n', auc_baselines_ranking, '\n', auc_methods_baselines_ranking)
    ''' Does image blurriness matter? '''
    laplacian_per_label_correct = df.groupby(['correct', 'label']).laplacian_var.mean()

    ''' Rank baselines based on std of deletion scores '''
    groups = df.groupby(['baseline']).del_scores_pred
    baseline_del_scores_std = groups.apply(lambda x: np.std(np.stack(x.values), axis=0))
    # convert rows to numpy arrays
    # baseline_del_scores_std = baseline_del_scores_std.apply(lambda x: np.array(x.tolist()))

    ''' Plot baseline deletion scores stds (how consistent is the baseline) '''
    if False:
        baseline_del_score_means = baseline_del_scores_std.apply(lambda x: np.mean(x)).sort_values()
        for i, m in zip(baseline_del_scores_std.index, baseline_del_score_means):
            row_vals = baseline_del_scores_std.loc[i]
            plt.plot(row_vals, label=f'{i}: {m:.3f}')
        plt_legend('baseline: std', np.arange(len(baseline_del_score_means)))
        plt.title('Baseline deletion scores std')
        plt.xticks(range(len(percentages_kept)), percentages_kept)
        plt.xlabel('Percentage of pixels kept')
        plt.ylabel('Deletion scores std')
        plt.tight_layout()
        output_path = join(cam_dir, f'baseline_del_scores_std.{ext}')
        plt_save_close(output_path)
        # black has the lowest std, which is good.

    ''' Plot CAMs as used in the deletion metric. '''
    if False:
        row = df[df.label == 0].iloc[41]
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


    def mix_mask(img, mask, inside=1, outside=0.3):
        """ Highlight masked area in the image, darken the rest. """
        return img * outside + mask * img * (inside - outside)


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
        df = df_sample.copy()
        df['dropoff_idx'] = df.apply(lambda row: np.argmax(row.del_scores_pred < 0.5), axis=1)  # argmax -> first
        # if all are >= 0.5, then argmax returns 0, which is not desired
        df['dropoff_pct'] = pct_kept[df.dropoff_idx]
        df[df.dropoff_pct == 100] = 0

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
        paths_kept = df[df.dropoff_pct >= 80].path.unique()
        for path in paths_kept:
            df_path = df[(df.path == path) & (df.baseline == b)]
            row = df_path.iloc[0]
            idx = row.idx
            img = np.array(Image.open(path), dtype=float) / 255
            # get dropoff explanation masks
            expls = df_path.apply(lambda row: perturbation_masks(cv2.resize(row.cam_pred, img_shape), [row.dropoff_pct])[0][..., None], axis=1)
            # blend masks with image
            blends = [mix_mask(img, (1 - expl)) for expl in expls]
            titles = [f'image idx={idx}'] + [f'{row.method} {100 - row.dropoff_pct:d}%' for _, row in df_path.iterrows()]
            title = f'GT: {nums_to_names_unif[row.label_unif]} pred: {nums_to_names[row.pred]}'  # ground truth always as multi-class for clarity
            plot_many([img, *blends], titles=titles, title=title, output_path=join(dropoffs_dir, f'del_idx{idx}_pred{row.pred}_gt{row.label_unif}.{ext}'))
            s = input()

        titles = [f'{100 - row.dropoff_pct:d}%' for _, row in df_kept.iterrows()]
        imgs = [Image.open(row.path) for _, row in df_kept.iterrows()]
        imgs = [np.array(img, dtype=float) / 255 for img in imgs]
        imgs_masked = [(im * mask * visible + (1 - mask) * im * muted) for im, mask in zip(imgs, dropoff_masks)]
        plot_many(imgs_masked, titles=titles, title='Dropoff masks')
        mask_removed = [(im * mask * muted + (1 - mask) * 1 * im * visible) for im, mask in zip(imgs, dropoff_masks)]
        plot_many(mask_removed, titles=descs, title='Dropoff explanations',
                  output_path=join(cam_dir, f'dropoff_explanations.{ext}'))

    if False:
        # re-do everything per-sample
        good_explanations = []
        # the idea is good, but it would be useful to group the samples by method
        # and mark the method that caused the dropoff.

        dfc = df.copy()
        dfc['dropoff_idx'] = dfc.apply(lambda row: np.argmax(row.del_scores_pred < 0.5), axis=1)  # argmax -> first
        dfc['dropoff_pct'] = pct_kept[dfc.dropoff_idx]
        dfc_kept = dfc[(dfc.dropoff_pct >= 80) & (dfc.dropoff_pct != 100)]
        for _, row in dfc_kept.iterrows():
            dropoff_mask = perturbation_masks(cv2.resize(row.cam_pred, img_shape), [row.dropoff_pct])[0][..., None]
            pred_name_dropoff = nums_to_names[np.argmax(row.del_scores[row.dropoff_idx])]
            desc = f'{row.idx}: {100 - row.dropoff_pct}% {nums_to_names[row.pred]} -> {pred_name_dropoff}'
            img = np.array(Image.open(row.path), dtype=float) / 255
            img_kept = mix_mask(img, dropoff_mask)
            expl_removed = mix_mask(img, (1 - dropoff_mask))

            # plot_many([img, row.cam_pred, expl_removed], titles=['image', 'expl', 'removed'], title=desc)
            # good_explanations.append({'expl': expl_removed, 'title': desc, 'path': row.path, 'method': row.method})

            # plot_many(img_kept)

            plot_many(expl_removed, title=desc)
            s = input()

        # we are interested in the smallest explanations that already cause the dropoff.
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
            df_k = df_k.sort_values('auc')  # ascending
            worst = df_k.head(n_samples)
            best = df_k.tail(n_samples)
            wb = pd.concat([worst, best], ignore_index=True)
            imgs = [Image.open(row['path']) for _, row in wb.iterrows()]
            imgs = [np.array(img, dtype=float) / 255 for img in imgs]

            # keep only area highlighted by cam_pred
            cam_upsampled = wb.cam_pred.apply(lambda x: cv2.resize(x, img_shape))
            cam_upsampled = np.stack(cam_upsampled, dtype=float)[..., None]  # (n, h, w, 1)
            imgs_masked_by_cam = [im * cam for im, cam in zip(imgs, cam_upsampled)]

            # plot_many(imgs, titles=[f'{line_auc:.3f}' for line_auc in wb.auc])
            plot_many(imgs_masked_by_cam, titles=[f'{a:.3f}' for a in wb.auc], title=f'{key}: {k_val}')
            # deletion_per_image(wb.del_scores, labels=[f'{r.idx}: {r.auc:.3f}' for _, r in wb.iterrows()], title=f'- {m}')

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
        cams = np.stack(df_sample.cam.values)  # todo overwriting `cams` from above
        cams = cams[:, :, 0, ...]  # drop 1-channel dim  -- not applicable
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
                # todo normalize to [0, 1]
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

    ''' Per-Method Deletion Metric Plot '''
    if True:
        output_path = join(cam_dir, f'deletion_metric_methods.{ext}')
        deletion_metric_per(df, 'method', output_path, ranges=False)

    ''' Per-class Deletion Metric Plot'''
    if False:
        """ 
        1 plot per baseline, 1 line per class.  
        Compare ways of perturbing the image with different baselines.
        """
        for baseline in baselines_unique:
            df_base = df[df.baseline == baseline]
            output_path = join(cam_dir, f'deletion_metric_{baseline}_classes.{ext}')
            deletion_metric_per_pred(df_base, output_path, f' - {baseline}', ranges=False, sort_aucs=False)

    # Plotting images, turn off grid
    plt.rcParams['axes.grid'] = True
    # set default matplotlib imshow colormap
    plt.rcParams['image.cmap'] = 'viridis'

    ''' Per-image CAM - per-class heatmaps '''
    if False:
        for i in range(3):
            s = ds[i]
            idx = s['idx']
            entries = df[df.idx == idx]
            for _, df_row in entries.iterrows():  # when loading more methods, idx is not unique
                output_path = join(cam_dir, f'{df_row.method}_idx{idx}.{ext}')
                plot2x3(df_row, s['image'], output_path, **{'vmin': 0, 'vmax': 1, 'cmap': 'viridis'})
            else:
                print(f"sample idx={s['idx']} not found in df")

    ''' Average CAM by predicted category '''
    if False:
        cams_pred = cam_mean(cams, preds)
        output_path = join(cam_dir, f'avg_class_pred.{ext}')
        plot1x5(cams_pred, 'Average CAM per predicted class', output_path)
        plot_many(cams_pred, title='Average CAM per predicted class', titles=label_names, output_path=output_path)

    ''' Average CAM by ground-truth category '''
    if False:
        cams_gt = cam_mean(cams, labels)
        output_path = join(cam_dir, f'avg_class_label.{ext}')
        plot1x5(cams_gt, 'Average CAM per ground_truth class', output_path)

    ''' Average CAM per (predicted, ground-truth) category '''
    if False:
        # confusion matrix for CAMs
        cams_confmat = cam_mean_confmat(cams, labels, preds)
        output_path = join(cam_dir, f'avg_confmat.{ext}')
        plot5x5(cams_confmat, 'Average CAM Confusion Matrix', output_path)

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
