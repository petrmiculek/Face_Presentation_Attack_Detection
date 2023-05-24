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
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use('tkagg')
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
from model_util import load_model
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

''' Parsing Arguments '''
parser = argparse.ArgumentParser()  # description='Evaluate model on dataset, run explanation methods'
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=None)
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
# parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)',
#                     type=str, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
# parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, default=None)
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
parser.add_argument('-l', '--lime', help='generate LIME outputs', action='store_true')
parser.add_argument('-c', '--cam', help='generate CAM outputs', action='store_true')
parser.add_argument('-v', '--eval', help='run evaluation loop', action='store_true')
parser.add_argument('-e', '--emb', help='get embeddings', action='store_true')
parser.add_argument('-z', '--show', help='show outputs', action='store_true')


# ^ the 4 above (lime, cam, eval, emb) are so far considered exclusive, although not enforced to be so


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
            labels.append(label)  # check if label gets mutated by .to() ...
            paths.append(sample['path'])

            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            out = model(img)
            loss = criterion(out, label)
            ep_loss += loss.item()

            prediction_hard = torch.argmax(out, dim=1)
            # save prediction
            preds.append(prediction_hard.cpu().numpy())

    # loss is averaged over batch already, divide by number of batches
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


def get_preds(dataset_loader, split_name, output_dir, new=True, save=True):
    """
    Get predictions for a dataset split -- load from file or run eval loop.
    :param dataset_loader:
    :param split_name:
    :param output_dir:
    :param new: create new predictions (run eval loop)
    :param save: save newly created predictions
    :return:
    """
    path_save_preds = join(output_dir, f'preds_{split_name}.npy')
    path_save_labels = join(output_dir, f'labels_{split_name}.npy')
    path_save_paths = join(output_dir, f'paths_{split_name}.npy')

    if new or not os.path.isfile(path_save_preds) \
            or not os.path.isfile(path_save_paths) \
            or not os.path.isfile(path_save_labels):
        print(f'Evaluating on: {split_name} set')
        res, preds, labels, paths = eval_loop(dataset_loader)
        print_dict(res)
        if save:
            # save predictions to file
            save_i(path_save_preds, preds)
            save_i(path_save_labels, labels)
            save_i(path_save_paths, paths)
    else:
        # load predictions from file
        print(f'Loading {split_name} set results from file')
        preds = np.load(path_save_preds)
        labels = np.load(path_save_labels)
        paths = np.load(path_save_paths)

    return {'labels': labels, 'preds': preds, 'paths': paths}


def load_model_eval(model_name, num_classes, run_dir, device='cuda:0'):
    """ Load Model """
    model_name = model_name
    model, preprocess = load_model(model_name, num_classes)
    model.load_state_dict(torch.load(join(run_dir, 'model_checkpoint.pt'), map_location=device), strict=False)
    model.to(device)
    model.eval()
    # sample model prediction
    out = model(torch.rand(1, 3, 224, 224).to(device)).shape  # hardcoded input size will fail at some point
    # assert shape is (1, num_classes)
    assert out == (1, num_classes), f'Model output shape is {out}'

    return model, preprocess


# def main():
#     global model, device, criterion  # disable when not using def main
if __name__ == '__main__':
    """
    Evaluate model on dataset, run explanation methods
    
    Note:
    - preprocess is not applied in DataLoader, but "manually" in eval_loop/similar
    
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    output_dir = join(run_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)

    print('Loading model and setup from:', run_dir)

    ''' Arguments '''
    batch_size = args.batch_size if args.batch_size else config_dict['batch_size']
    num_workers = args.num_workers if args.num_workers is not None else config_dict['num_workers']
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
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, limit=limit, quiet=False, **loader_kwargs)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified

    t0 = time.perf_counter()

    ''' LIME - Generate explanations for dataset split/s '''
    if args.lime:
        print('Generating LIME explanations')
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        import matplotlib.pyplot as plt


        def convert_for_lime(img):
            return (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


        def predict_lime(images):
            """
            Run prediction on an image
            uses global model and device

            - check for batch size > 1  #DONE#
            - check for uint8 => divide by 255 again  #DONE#
            :param images: HWC numpy array
            :return: softmax-ed probabilities
            """
            img0 = torch.tensor(images)
            if len(img0.shape) == 3:
                img0 = img0.unsqueeze(0)

            if img0.dtype == torch.uint8:
                img0 = img0.float() / 255.

            img0 = img0.permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                logits = model(img0)
                probs = F.softmax(logits, dim=1)
                res = probs.cpu().numpy()
            return res


        def show_lime_image(explanation, img, lime_kwargs, title=None, show=False, output_path=None):
            """
            Show image with LIME explanation overlay

            todo parameterize label index [opt]
            """
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], **lime_kwargs)
            half_overlay = temp // 2 + img // 2
            img_lime_out = mark_boundaries(half_overlay / 255.0, mask)
            plt.imshow(img_lime_out)
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            if show:
                plt.show()

            if output_path:
                plt.savefig(output_path)

            plt.close()


        nums_to_names = dataset_module.nums_to_unified

        explainer = lime_image.LimeImageExplainer(random_state=seed)

        labels = []
        paths = []
        preds = []
        idxs = []

        lime_dir = join(output_dir, 'lime')
        os.makedirs(lime_dir, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(test_loader, mininterval=1., desc='Eval'):
                img_batch, label = batch['image'], batch['label']
                idx = batch['idx']
                labels.append(label)  # check if label gets mutated by .to() ...
                paths.append(batch['path'])
                idxs.append(idx)
                for i, img in tqdm(enumerate(img_batch), mininterval=1., desc='\tBatch', leave=False,
                                   total=len(img_batch)):
                    img_for_lime = convert_for_lime(img)
                    explanation = explainer.explain_instance(img_for_lime, predict_lime, batch_size=16,
                                                             top_labels=1, hide_color=0, num_samples=1000,
                                                             progress_bar=False, random_seed=seed)

                    pred_top1 = explanation.top_labels[0]
                    preds.append(pred_top1)
                    pred_top1_name = nums_to_names[pred_top1]

                    label_name = label_names[label[i]]

                    # positive-only
                    title = f'LIME explanation (pos), pred {pred_top1_name}, GT {label_name}'
                    lime_kwargs = {'positive_only': True, 'num_features': 5, 'hide_rest': False}
                    output_path = join(lime_dir, f'{idx[i]}_pos.png')

                    show_lime_image(explanation, img_for_lime, lime_kwargs, title, output_path=output_path)
                    # show_lime_image(explanation, img_for_lime, lime_kwargs, title, show=True)

                    # positive and negative
                    title = f'LIME explanation (pos+neg), pred {pred_top1_name}, GT {label_name}'
                    lime_kwargs = {'positive_only': False, 'num_features': 10, 'hide_rest': False}
                    output_path = join(lime_dir, f'{idx[i]}_pos_neg.png')

                    show_lime_image(explanation, img_for_lime, lime_kwargs, title, output_path=output_path)
                    # show_lime_image(explanation, img_for_lime, lime_kwargs, title, show=True)

        labels = torch.cat(labels).numpy()
        idxs = torch.cat(idxs).numpy()
        paths = np.concatenate(paths)
        preds = np.array(preds)
        idxs = np.array(idxs)

        np.savez(join(lime_dir, 'lime.npz'), labels=labels, paths=paths, preds=preds, idxs=idxs)
        print(f'LIME explanations saved to {lime_dir}')

    ''' Evaluation '''
    if args.eval:
        ''' Training set '''
        # outputs_train = get_preds(train_loader, 'train', output_dir, new=True, save=True)

        ''' Validation set '''
        # outputs_val = get_preds(val_loader, 'val', output_dir, new=True, save=True)

        ''' Test set '''
        outputs_test = get_preds(test_loader, 'test', output_dir, new=True, save=True)

        metrics_test = compute_metrics(outputs_test['labels'], outputs_test['preds'])
        metrics_test = keys_append(metrics_test, ' Test')

        cr = classification_report(outputs_test['labels'], outputs_test['preds'])
        print(cr)

        ''' Confusion matrix '''
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
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
        FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if args.cam:
        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)
        target_layers = model.features[-1]  # [model.layer4[-1]]  # resnet18
        grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputTarget(cat) for cat in range(config_dict['num_classes'])]

        cams_out = []  # list of dicts: { idx: int, label: int, pred: int, path: str, cam: np.array[C, H, W] }
        for batch in tqdm(test_loader, mininterval=2., desc='CAM'):
            ''' Predict (batch) '''
            with torch.no_grad():
                preds_raw = model(batch['image'].to(device)).cpu()
                preds = F.softmax(preds_raw, dim=1).numpy()
                preds_classes = np.argmax(preds, axis=1)

            ''' Generate CAMs for images in batch '''
            for i, img in enumerate(batch['image']):
                pred = preds[i]
                idx = batch['idx'][i].item()
                label = batch['label'][i].item()
                cams = []
                for t in targets:
                    grayscale_cam = grad_cam(input_tensor=img[None, ...], targets=[t])  # img 4D
                    grayscale_cam = grayscale_cam[0, ...]  # -> 3D
                    cams.append(grayscale_cam)

                cams = np.stack(cams)  # [C, H, W]
                # cast cams to uint8  # note: attempt to save space
                cams = (255 * cams).astype(np.uint8)

                cams_out.append({'cam': cams,
                                 'idx': idx,
                                 'label': label,
                                 'path': batch['path'][i],  # strings are not tensors
                                 'pred': preds_classes[i],
                                 })
            # end of batch
        # end of dataset

        ''' Save CAMs '''
        cams_df = pd.DataFrame(cams_out)
        cams_df.to_pickle(join(cam_dir, 'cams.pkl.gz'), compression='gzip')
        if False:
            # from src.util import dol_from_lod
            # cams_out = dol_from_lod(cams_out)
            #
            # cams_objs = [c['cam'] for c in cams_out]
            # cams_objs = np.stack(cams_objs)  # [N, C, H, W] = [N, 5, 12, 12]
            # idxs = [c['idx'] for c in cams_out]
            # # save compressed numpy array
            # np.savez_compressed(join(cam_dir, 'cams.npz'), cam=cams_objs, idx=idxs)
            pass

        if False:
            ''' Average CAM per target class '''
            canvas = None
            target = 0
            for c in cams_out:
                cam_i = c['cam'][target]
                if canvas is None:
                    canvas = cam_i
                else:
                    canvas += cam_i

            canvas /= len(cams_out)

            plt.imshow(canvas)
            # colorbar
            plt.colorbar()
            plt.show()

    ''' Old CAM generation with overlayed visualization '''
    # new approach (above) is to just generate-save, and analyse later
    # todo copy CAM methods to new approach [func]
    if False:
        methods = [GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad]

        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)

        target_layers = model.features[-1]  # [model.layer4[-1]]  # resnet18

        labels_list = []
        paths_list = []
        preds_list = []
        idxs_list = []
        cams_all = dict()

        nums_to_names = dataset_module.nums_to_unified

        ''' Iterate over batches in dataset '''
        for batch in tqdm(test_loader, mininterval=2., desc='CAM'):
            with torch.no_grad():
                preds_raw = model(batch['image'].to(device)).cpu()
                preds = F.softmax(preds_raw, dim=1).numpy()
                preds_classes = np.argmax(preds, axis=1)

            labels_list.append(batch['label'])
            paths_list.append(batch['path'])
            idxs_list.append(batch['idx'])
            ''' Iterate over images in batch '''
            for i, img in enumerate(batch['image']):
                # tqdm(..., mininterval=2., desc='\tBatch', leave=False, total=len(img_batch)):

                pred = preds[i]
                idx = batch['idx'][i].item()
                label = batch['label'][i].item()

                img_np = img.cpu().numpy().transpose(1, 2, 0)  # img_np 3D

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

                        # rescale from [min, max] to [0, 1]
                        img_np_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                        # note: with the edited library code giving true CAM sizes (not input image size), you have to rescale manually
                        # this is not done here as of now
                        visualization = show_cam_on_image(img_np_norm, grayscale_cam, use_rgb=True)
                        cams.append(grayscale_cam)
                        overlayed.append(visualization)

                    img_cams[method_name] = {'cams': cams, 'overlayed': overlayed}

                    if False:
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
                if False:
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

    ''' Embeddings '''
    if args.emb:
        """
        A) Through forward hooks
        - no necessity to modify model
        B) Through forward method
        - avoids __call__ method, might ignore other hooks
        
        Both ways currently work: either extracting it as a hook or through a special forward method
        
        """

        # if not hasattr(model, 'fw_with_emb'):
        #     raise AttributeError('Model does not have fw_with_emb method')  # currently only for resnet18

        ''' Extract embeddings through forward hooks '''
        activation = {}


        def get_activation(name):
            """ Hook for extracting activations from a layer specified by name """

            def hook(model, inputs, outputs):
                activation[name] = outputs.detach()

            return hook


        emb_layer = 'avgpool'  # resnet18
        # emb_layer = 'avgpool'  # efficientnet_v2_s
        embeddings_hook = model.avgpool.register_forward_hook(get_activation(emb_layer))

        ''' Eval loop '''
        len_loader = len(test_loader)
        ep_loss = 0.0
        avgpools = []
        preds = []
        labels = []
        paths = []
        features = []
        # todo: this could be unified with the eval_loop, just add the fw_hook there and save its output [clean]
        with torch.no_grad():
            for sample in tqdm(test_loader, mininterval=1., desc='Embeddings'):
                img, label = sample['image'], sample['label']
                labels.append(label)  # check if label gets mutated by .to() ...
                paths.append(sample['path'])

                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
                # out, feature = model.fw_with_emb(img)
                out = model(img)

                loss = criterion(out, label)
                ep_loss += loss.item()

                prediction_hard = torch.argmax(out, dim=1)
                preds.append(prediction_hard.cpu().numpy())

                avgpools.append(activation['avgpool'].cpu().numpy())
                # features.append(feature.cpu().numpy())

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        paths = np.concatenate(paths)
        avgpools = np.concatenate(avgpools)[..., 0, 0]  # strip last 2 dimensions (N, C, 1, 1)
        # features = np.concatenate(features)

        embeddings_hook.remove()

        # save avgpools embeddings to file
        split = 'test'
        np.save(join(output_dir, f'embeddings_{split}.npy'), avgpools)

        # loss is averaged over batch already, divide by number of batches
        ep_loss /= len_loader

        metrics = compute_metrics(labels, preds)

        features = avgpools  # using the forward hooks, not fw_with_emb

        # embeddings distance matrix
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances

        # cosine similarity
        cos_sim = cosine_similarity(features)
        # euclidean distance
        eucl_dist = euclidean_distances(features)

        """
        Note:
        for a 7k dataset, cosine similarity takes 1.5s, euclidean distance 2.5s
        for a 50k+ dataset, we might wait a bit
        """

    t1 = time.perf_counter()
    print(f'Execution finished in {t1 - t0:.2f}s')  # everything after dataset is loaded

    ''' Sandbox Area'''
    aug = False
    if aug:
        # call for each image in batch separately or for the whole batch?
        ''' Unused '''
        """
        # for a runnable augmentation example look into `augmentation.py`
        
        # set default print float precision
        np.set_printoptions(precision=2, suppress=True)

        import torchvision.transforms.functional as F
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # what are the sample values (min, max, avg, std) for each?
        for i, img in enumerate([sample['image'], img_pre, img_aug, img_aug_pre]):
            img_n = F.normalize(img, mean=mean, std=std)
            for j, img_v in enumerate([img, img_n]):

                if False:
                    v = img_v.numpy()
                    v_min = v.min(axis=(0, 2, 3))
                    v_max = v.max(axis=(0, 2, 3))
                    v_mean = v.mean(axis=(0, 2, 3))
                    v_std = v.std(axis=(0, 2, 3))
                    print(f'[{i}, {j}]:\t'
                          f'Min: {v_min},\t'
                          f'Max: {v_max},\t'
                          f'Mean: {v_mean},\t'
                          f'Std: {v_std}')

        # out, feature = model.fw_with_emb(img_batch)
        out_raw = model(sample['image'].to(device))
        out_pre = model(img_pre.to(device))
        out_aug = model(img_aug.to(device))
        out_aug_pre = model(img_aug_pre.to(device))

        for o in [out_raw, out_pre, out_aug, out_aug_pre]:
            loss = criterion(o, sample['label'].to(device))
            print(f'Loss: {loss.item():.4e}')
        """

        """
        How to do with augmentation, preprocessing, and collate_fn:
        - preprocess not used at all
        - collate_fn makes everything a tensor, it runs in the worker processes, I do not provide it (kept default)
        - transform 
            - transform=augmentation -> dataset loader -> dataset constructor, and runs in __getitem__  (I think in the worker processes)
            - I need to provide both the augmentation functions for train and eval
            - 
        - augmentation can also run in training loop in main process
        
        how to make it work:
        - first, don't waste time with transforms, just do it in the training loop
        - then, make it work with transforms
        
        - ignore preprocess (don't run it pls)
        - keep collate_fn as is and transforms empty
        - add augmentation to the training loop
        - make sure that eval uses the eval "aug"
        
        - first verify in eval
            - non-augmented image processing should be equivalent
        
        """
