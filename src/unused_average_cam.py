"""
I need:
model (runs/dir)
predictions
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

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
# import dataset_rose_youtu as dataset
from metrics import compute_metrics, confusion_matrix  # , accuracy
from util import print_dict, save_i, keys_append
from util_torch import load_model_eval
import config

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

transform_train = None
transform_eval = None
''' Global variables '''
# -

parser = argparse.ArgumentParser(description='...')

if __name__ == '__main__':
    """
    Evaluate model on dataset, run explanation methods
    
    Note:
    - preprocess is not applied in DataLoader, but "manually" in eval_loop/similar
    
    """
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    output_dir = join(run_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)

    print('Loading model and setup from:', run_dir)

    ''' Arguments '''
    batch_size = args.batch_size if args.batch_size else config_dict['batch_size']
    num_workers = args.num_workers if args.num_workers else config_dict['num_workers']
    limit = args.limit if args.limit else -1
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
    # device = torch.device("cpu")
    print(f'Running on device: {device}')
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    seed = args.seed if args.seed else config.seed_eval_default
    print(f'Random seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    model, preprocess = load_model_eval(config_dict['model_name'], config_dict['num_classes'], run_dir)
    print(f'Model: {config_dict["model_name"]} with {config_dict["num_classes"]} classes')

    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_dataset

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                         'seed': seed, 'drop_last': False,
                         'transform_train': preprocess['eval'], 'transform_eval': preprocess['eval']}
        #                                               ^^^^ note: eval transform is used for both train and test
        path_prefix = ...  # TODO set path_prefix
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, path_prefix=path_prefix, limit=limit, quiet=False,
                         **loader_kwargs)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified

    t0 = time.perf_counter()
    if args.cam:
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
            # img_batch, label = batch['image'], batch['label']
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

                    # todo methods could be loaded outside the loop [clean]
                    method_name = method.__name__
                    grad_cam = method(model=model, target_layers=target_layers, use_cuda=True)

                    targets = [ClassifierOutputTarget(cat) for cat in range(config_dict['num_classes'])]

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

                    if True:
                        ''' Plot CAMs '''
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

        labels_list = np.concatenate(labels_list)
        paths_list = np.concatenate(paths_list)
        idxs_list = np.concatenate(idxs_list)

        ''' Save CAMs npz '''
        maybe_limit = f'_limit{limit}' if limit else ''
        path = join(cam_dir, f'cams_{dataset_name}{maybe_limit}.npz')
        np.savez(path, cams_all=cams_all, labels=labels_list, paths=paths_list, idxs=idxs_list)
        print(f'Saved CAMs to {path}')
