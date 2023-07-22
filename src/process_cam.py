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
from dataset_base import show_labels_distribution, pick_dataset_version, load_annotations
from util_image import overlay_cam, get_marker, plot_many, sobel_edges
from util_torch import init_seed, get_dataset_module
from util_face import get_ref_landmarks

''' Global variables '''
run_dir = ''
nums_to_names = None
label_names = None
args = None
cam_dir = None
ext = 'png'  # extension for saving plots
percentages_kept = None  # deletion metric percentages [0, 100]
auc_percentages = None

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


def plot1x5(cams, title='', output_path=None):
    """Plot 5 CAMs in a row."""
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
    plt_save_close(output_path)


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
    plt_save_close(output_path)


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
    plt_save_close(output_path)


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


def deletion_per_image(del_scores, line_labels=None, many_ok=False, title=None, output_path=None):
    # x: perturbation level, y: prediction drop
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


def auc(deletion_scores, percentages=None):
    """ Compute the Area under Curve (AUC) for a given deletion curve. """
    if percentages is None:
        percentages = auc_percentages
    auc_ = np.trapz(deletion_scores, -percentages)
    # using negative percentages, because trapz() assumes increasing x-values
    return auc_


def deletion_metric_per(df, key, output_path=None, line_labels=None, ranges=True, sort_aucs=True):
    key_unique = df[key].unique()
    aucs = []
    for k_val in key_unique:
        df_kval = df[df[key] == k_val]
        del_scores = np.stack(df_kval.del_scores_pred.values)
        ys = np.mean(del_scores, axis=0)
        auc_mean = df_kval.auc.mean()
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
    order = np.argsort(aucs)
    plt_legend(f'{key}: AUC', order, line_labels)
    plt.tight_layout()
    plt_save_close(output_path)


def deletion_metric_per_pred(df_base, output_path=None, title=None, ranges=True, sort_aucs=True):
    """ Plot Deletion metric by predicted class, """
    del_scores = df_base['del_scores_pred'].values
    preds = df_base['pred'].values
    aucs = []
    for i, label in enumerate(label_names):
        y_pred_class = del_scores[preds == i]
        if len(y_pred_class) == 0:
            continue
        y_pred_class = np.stack(y_pred_class)
        mean_y = np.mean(y_pred_class, axis=0)
        auc_mean = auc(mean_y)
        aucs.append(auc_mean)
        # plot mean + std
        plt.plot(percentages_kept, mean_y, label=f'{label}: {auc_mean:.3f}')
        if ranges:
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


def rank_by_auc(df, columns):
    return df.groupby(columns)['auc'].mean().sort_values(ascending=False)


# def main():
if __name__ == '__main__':
    # global auc_percentages, percentages_kept  # only uncomment when using def main()
    """
    Process CAMs - perform analysis, visualize, and save plots.
    """
    ''' Arguments '''
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
    # eval directory name format to parse, hyphen-separated:
    # 'eval', dataset_name, dataset_note [, attack], mode, limit_
    _, dataset_name, dataset_note, mode, *attack, limit_ = eval_dir.split('-')
    attack = attack[0] if len(attack) == 1 else None

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
    df_orig = df.copy()
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
        split = 'test'  # implicitly always working with unseen data
        annotations = load_annotations(dataset_meta, seed, args.path_prefix, limit)[split]
        ds = dataset_module.Dataset(annotations)
        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified
        num_classes = dataset_meta['num_classes']
        nums_to_names = dataset_module.nums_to_unified
        ''' Show labels distribution '''
        show_labels_distribution(annotations['label'], split, num_classes)

    ''' Metadata '''
    df['shp'] = df.cam.apply(lambda x: x.shape)
    # add metadata from annonations file
    df = df.merge(annotations, on='idx', how='inner', suffixes=('', '_y'))

    # Check CAMs shapes
    if len(df.shp.unique()) != 1:
        raise ValueError(f'Different CAM shapes!\n{df.shp.value_counts()}')
    cam_shape = df.iloc[0]['shp']
    dg = df[['method', 'shp']].groupby('shp')
    # which shapes are there for which methods?
    for g, rows in dg:
        print(f'{g}:\n', rows.method.value_counts())

    if len(dg) != 1:
        filter_idxs = df.shp == cam_shape
        print(f'Filtering CAMs to keep only the first shape {sum(filter_idxs)}/{len(filter_idxs)}.')
        df = df[df.shp == cam_shape]
    print(f'CAM shape: {cam_shape}')

    df_sample = df.sample(10)
    dfcopy = df.copy()
    # df = df_sample  # TODO TEMPORARY

    ''' AUC for each CAM '''
    percentages_kept = df['percentages_kept'].values[0]  # assume same for all
    auc_percentages = np.array(percentages_kept) / 100  # normalize to [0, 1]
    # deletion scores for predicted class
    df['del_scores_pred'] = df.apply(lambda row: row.del_scores[:, row.pred], axis=1)
    df['auc'] = df['del_scores_pred'].apply(auc)

    df['correct'] = df['label'] == df['pred']
    df.cam = df.cam.apply(lambda x: np.float32(x) / 255)  # from uint8 to [0, 1] float
    df['cam_pred'] = df.apply(lambda row: row.cam[row.pred], axis=1)  # H x W
    cams = np.stack(df['cam'].values)
    labels = df['label'].values
    preds = df['pred'].values
    idxs = df['idx'].values
    methods_unique = df['method'].unique()
    baselines_unique = df['baseline'].unique()
    sample_row = df.iloc[0]
    sample_img = Image.open(sample_row.path)
    img_shape = sample_img.size  # (W, H), assume same for all
    landmarks = get_ref_landmarks()
    ###############################################################################
    ''' ------- Exploring CAMs ------- '''
    # First plot charts, keep grid on. (Then turn it off for images)
    plt.rcParams['axes.grid'] = True

    ''' Rank by AUC '''
    auc_methods_ranking = rank_by_auc(df, ['method'])
    auc_baselines_ranking = rank_by_auc(df, ['baseline'])
    auc_methods_baselines_ranking = rank_by_auc(df, ['method', 'baseline'])
    print(auc_methods_ranking, '\n', auc_baselines_ranking, '\n', auc_methods_baselines_ranking)
    ''' Does image blurriness matter? '''
    laplacian_per_label_correct = df.groupby(['correct', 'label']).laplacian_var.mean()

    ''' Rank baselines based on std of deletion scores '''
    groups = df.groupby(['baseline']).del_scores_pred  # .std().sort_values(ascending=False)
    baseline_del_scores_std = groups.apply(lambda x: np.std(np.stack(x.values), axis=0))
    # convert rows to numpy arrays
    # baseline_del_scores_std = baseline_del_scores_std.apply(lambda x: np.array(x.tolist()))

    ''' Plot baseline stds '''
    row_means = baseline_del_scores_std.apply(lambda x: np.mean(x)).sort_values()
    for i, m in zip(baseline_del_scores_std.index, row_means):
        row_vals = baseline_del_scores_std.loc[i]
        plt.plot(row_vals, label=f'{i}: {m:.3f}')
    plt_legend('baseline: std', np.arange(len(row_means)))
    plt.title('Baseline deletion scores std')
    plt.xticks(range(len(percentages_kept)), percentages_kept)
    plt.xlabel('Percentage of pixels kept')
    plt.ylabel('Deletion scores std')
    plt.tight_layout()
    plt.show()
    # black has the lowest std, which is good.
    ''' Does baseline choice matter? '''  # (older note)
    if False:
        pass
        # average AUC
        # difference between del_scores across baselines

    ''' Plot CAMs as used in the deletion metric. '''
    if False:
        row = df[df['label'] == 3].iloc[42]
        method = row['method']
        pred_class = row['pred']
        cam_pred = (row['cam'][pred_class][0]).astype(float) / 255  # -> H x W = 12, 12
        cam_pred_copy = cam_pred.copy()
        img_pil = Image.open(join(args.path_prefix, row['path']))
        img = np.array(img_pil, dtype=float) / 255  # HWC
        # resize to original image size
        wh_image = (img.shape[0], img.shape[1])
        if cam_pred.shape != wh_image:
            cam_pred = cv2.resize(cam_pred, wh_image)  # interpolation bilinear by default

        ''' Perturbation baselines '''
        b_black = np.zeros_like(cam_pred)
        b_blur = np.dstack(  # here dstack for HWC
            [cv2.blur(img[..., c], (config.blur_img_s, config.blur_img_s)) for c in range(3)])
        b_mean = np.zeros_like(img) + np.mean(img, axis=(0, 1), keepdims=True)
        b_blurdark = (b_black[..., None] + b_blur) / 4
        plot_many([img, b_black, b_blur, b_mean, b_blurdark], title='Perturbation baselines',
                  titles=['image', 'black', 'blur', 'mean', 'blurdark'])

        plot_many([img, cam_pred_copy, cam_pred], title=f'{row.idx} {method}',
                  titles=['image', 'CAM', 'CAM upsampled to image'])
        # low and high resolution
        ''' Perturb explained region '''
        cam_blur_weight = config.cam_blurred_weight
        cam_blurred = cam_pred + cam_blur_weight * cv2.blur(cam_pred, (config.blur_cam_s, config.blur_cam_s))
        thresholds = np.percentile(cam_blurred, percentages_kept)
        thresholds[-1] = cam_blurred.min()  # make sure 0% kept is 0

        plot_many([cam_pred, cam_blurred], titles=['cam_pred', 'cam_blurred'])

        baseline_name = row['baseline']
        baseline = b_mean
        imgs_perturbed = []
        masks = []
        for th, pct in zip(thresholds, percentages_kept):
            mask = (cam_blurred < th)[..., None]
            img_masked = (img * mask + (1 - mask) * baseline)
            # img_masked_plotting = img_plotting * mask.cpu().numpy()
            imgs_perturbed.append(img_masked)
            masks.append(mask)

        plot_many(imgs_perturbed, titles=percentages_kept, title=f'{cam_blur_weight=}')
        plot_many(masks, title='masks', titles=percentages_kept)

        # showerthought: why do I compute the sobel edges for the full-size image?
        # do it at the CAM size. No, it really looks more credible for full-size.
        cam_wh = cam_pred_copy.shape

        sobel_full = sobel_edges(img)
        sobel_downsampled = cv2.resize(sobel_full, cam_wh, interpolation=cv2.INTER_AREA)

        img_camsized = cv2.resize(img, cam_wh)
        sobel_camsized = sobel_edges(img_camsized)

        plot_many([img, sobel_full, img_camsized, sobel_camsized, sobel_downsampled])

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
            cam_pred_upsampled = wb.cam_pred.apply(lambda x: cv2.resize(x, img_shape))
            cam_pred_upsampled = np.stack(cam_pred_upsampled, dtype=float) / 255
            imgs_masked_by_cam = []
            for im, cam in zip(imgs, cam_pred_upsampled):
                masked = im * cam[..., None]
                imgs_masked_by_cam.append(masked)

            # plot_many(imgs, titles=[f'{a:.3f}' for a in wb.auc])
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
        deletion_per_image(df10.del_scores, df10.idx, output_path=output_path)

    ''' Per-Method Deletion Metric Plot '''
    if True:
        output_path = join(cam_dir, f'deletion_metric_methods.{ext}')
        deletion_metric_per(df, 'method', output_path, ranges=True)

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

    ''' Per-image CAM - per-class heatmaps '''
    if False:
        for i in range(3):
            s = ds[i]
            idx = s['idx']
            entries = df[df.idx == idx]
            for _, cam_entry in entries.iterrows():  # when loading more methods, idx is not unique
                cam_method_name = cam_entry['method']
                output_path = join(cam_dir, f'{cam_method_name}_idx{idx}.{ext}')
                plot2x3(cam_entry, s['image'], output_path, **{'vmin': 0, 'vmax': 255, 'cmap': 'viridis'})
            else:
                print(f"sample idx={s['idx']} not found in df")

    ''' Average CAM by predicted category '''


    def cam_mean(cams, select):
        cam = np.zeros(cam_shape)  # (c, h, w)
        # filter all RuntimeWarning
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            for i in range(len(label_names)):
                cam[i] = np.mean(cams[select == i], axis=0)[i]
        cam[np.isnan(cam)] = 0
        return cam


    cams_id = 'SampleID'
    if False:
        cams_pred = cam_mean(cams, preds)
        output_path = join(cam_dir, f'{cams_id}_avg_class_pred.{ext}')
        plot1x5(cams_pred, 'Average CAM per predicted class', output_path)

    ''' Average CAM by ground-truth category '''
    if False:
        cams_gt = cam_mean_per(cams, labels)
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
    n_incorrect = len(df) - df.correct.sum()
    if False and n_incorrect > 0:
        print(f'Incorrect predictions: {n_incorrect} / {len(df)}')
        limit_incorrect = 20
        if n_incorrect > limit_incorrect:
            print(f'Too many incorrect predictions to plot, limiting to {limit_incorrect}')
        n_plotted = 0
        for i, row in df.iterrows():
            if row['pred'] == row['label']:
                continue
            if n_plotted >= limit_incorrect:
                break
            n_plotted += 1
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

    # unused - copied from evaluate.py
    if False:
        cams_df = None
        # plot deletion scores per-baseline
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
