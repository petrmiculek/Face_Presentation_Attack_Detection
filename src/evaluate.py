# stdlib
import argparse
import logging
import os
import time
from os.path import join

# fix for local import problems - add all local directories
import sys

from sklearn.metrics import classification_report

sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# external

from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

import numpy as np
import os
import json

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
from model_util import load_model

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
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
# parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)',
#                     type=str, default=None)
# parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, default=None)
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', type=str, default=None)
parser.add_argument('-l', '--lime', help='generate LIME outputs', action='store_true')
parser.add_argument('-c', '--cam', help='generate CAM outputs', action='store_true')
parser.add_argument('-v', '--eval', help='run evaluation loop', action='store_true')
parser.add_argument('-e', '--emb', help='get embeddings', action='store_true')
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)


# ^ the 4 above are so far considered exclusive, although not enforced to be so


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
            img = preprocess(img)  # todo check ok after preprocess re-added here (ditto other occurences) [func]
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
    path_save_preds = join(output_dir, f'preds_{split_name}.npy')
    path_save_labels = join(output_dir, f'labels_{split_name}.npy')
    path_save_paths = join(output_dir, f'paths_{split_name}.npy')

    if new or not os.path.isfile(path_save_preds) \
            or not os.path.isfile(path_save_paths) \
            or not os.path.isfile(path_save_labels):
        print(f'{split_name} set')
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
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
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

    model, preprocess = load_model_eval(config_dict['model_name'], config_dict['num_classes'], run_dir)

    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_dataset

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                         'drop_last': False}  # , 'transform': preprocess <- not used on purpose
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, limit=limit, quiet=False, **loader_kwargs)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified

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
                img0 = preprocess(img0)  # not necessary since loader preprocesses
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

        explainer = lime_image.LimeImageExplainer()

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
                                                             progress_bar=False)

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
        print('Training set')
        outputs_train = get_preds(train_loader, 'train', output_dir)

        print('Validation set')
        outputs_val = get_preds(val_loader, 'val', output_dir)

        print('Test set')
        outputs_test = get_preds(test_loader, 'test', output_dir, new=True, save=False)

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
    if args.cam:
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            FullGrad
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import seaborn as sns

        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)

        target_layers = [model.layer4[-1]]

        sample = next(iter(train_loader))
        imgs, labels = sample['image'], sample['label']
        with torch.no_grad():
            preds_raw = model.forward(imgs.to(device)).cpu()
            preds = F.softmax(preds_raw, dim=1).numpy()
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
                    plt.title(label_names[j] + label_pred_score + matches_label)
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

        gt_label_name = label_names[label_scalar]
        pred_label_name = label_names[pred.argmax()]

        j = 0
        for name, cs in method_cams_dict.items():
            c = cs['overlayed'][label_scalar]
            plt.subplot(3, 3, j + 2)
            plt.imshow(c)
            label_pred_score = f': {preds[i, j]:.2f}'  # type: ignore
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

    ''' Embeddings '''
    if args.emb:
        """
        A) Through forward hooks
        - no necessity to modify model
        B) Through forward method
        - avoids __call__ method, might ignore other hooks
        
        Both ways currently work: either extracting it as a hook or through a special forward method
        
        """

        if not hasattr(model, 'fw_with_emb'):
            raise AttributeError('Model does not have fw_with_emb method')  # currently only for resnet18

        ''' Extract embeddings through forward hooks '''
        activation = {}


        def get_activation(name):
            """ Hook for extracting activations from a layer specified by name """

            def hook(model, inputs, outputs):
                activation[name] = outputs.detach()

            return hook


        embeddings_hook = model.avgpool.register_forward_hook(get_activation('avgpool'))

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
                img_batch = preprocess(img)
                out, feature = model.fw_with_emb(img_batch)

                loss = criterion(out, label)
                ep_loss += loss.item()

                prediction_hard = torch.argmax(out, dim=1)
                preds.append(prediction_hard.cpu().numpy())

                avgpools.append(activation['avgpool'].cpu().numpy())
                features.append(feature.cpu().numpy())

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        paths = np.concatenate(paths)
        avgpools = np.concatenate(avgpools)[..., 0, 0]  # strip last 2 dimensions (N, C, 1, 1)
        features = np.concatenate(features)

        embeddings_hook.remove()

        # save avgpools embeddings to file
        split = 'test'
        np.save(join(output_dir, f'embeddings_{split}.npy'), avgpools)

        # loss is averaged over batch already, divide by number of batches
        ep_loss /= len_loader

        metrics = compute_metrics(labels, preds)

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
