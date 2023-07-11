# stdlib
import argparse
import logging
import os
import json
import sys
import time
from os.path import join

# fix for local import problems
sys.path.append(os.getcwd())  # + [d for d in os.listdir() if os.path.isdir(d)]

# external
from sklearn.metrics import classification_report
from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
# matplotlib.use('tkagg')  # helped for some plotting issues (local console run, does not work on metacentrum)
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
# import dataset_rose_youtu as dataset
from metrics import compute_metrics, confusion_matrix  # , accuracy
from util import print_dict, save_i, keys_append, plot_many
from util_torch import init_device, init_seed, load_model_eval, get_dataset_module
from dataset_base import pick_dataset_version, load_dataset
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

''' Global variables '''
# -

''' Parsing Arguments '''
parser = argparse.ArgumentParser()  # description='Evaluate model on dataset, run explanation methods'
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=None)
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
parser.add_argument('-p', '--path', help='path to dataset', type=str, default=None)

def predict(model, inputs):
    """ Predict on batch, return Numpy preds and classes. """
    with torch.no_grad():
        preds_raw = model(inputs)
        probs = F.softmax(preds_raw, dim=1).cpu().numpy()
        classes = np.argmax(probs, axis=1)
    return probs, classes

def eval_loop(loader):
    """ Evaluate model on dataset. """
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



# def main():
#     global model, device, criterion  # disable when not using def main
if __name__ == '__main__':
    """
    Evaluate model on dataset, run explanation methods
    Note:
    - preprocess is obtained automatically in load_model, and applied in DataLoader. Do not use it manually.
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    if sum([args.lime, args.cam, args.eval, args.emb]) > 1:
        raise ValueError('Only one of lime, cam, eval, emb can be set')

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
    dataset_module = get_dataset_module(dataset_name)

    ''' Initialization '''
    device = init_device()
    seed = args.seed if args.seed is not None else config.seed_eval_default
    init_seed(seed)

    model, preprocess = load_model_eval(config_dict['model_name'], config_dict['num_classes'], run_dir, device)
    print(f'Model: {config_dict["model_name"]} with {config_dict["num_classes"]} classes')
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                         'seed': seed, 'drop_last': False,
                         'transform_train': preprocess['eval'], 'transform_eval': preprocess['eval']}
        #                                               ^^^^ note: eval transform is used for both train and test
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, path_prefix=args.path,
                         limit=limit, quiet=False, **loader_kwargs)

        bona_fide = dataset_module.bona_fide
        label_names = dataset_module.label_names_unified

    t0 = time.perf_counter()

    ''' LIME Explanations '''
    if args.lime:
        print('Generating LIME explanations')
        from lime import lime_image
        from skimage.segmentation import mark_boundaries


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
            if len(img0.shape) == 3:  # img0.ndim
                img0 = img0.unsqueeze(0)

            if img0.dtype == torch.uint8:
                img0 = img0.float() / 255.

            img0 = img0.permute(0, 3, 1, 2).to(device)

            probs, _ = predict(model, img0)
            return probs


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
                labels.append(label)
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
                    # show_lime_image(explanation, img_for_lime, lime_kwargs, title, show=args.show)

                    # positive and negative
                    title = f'LIME explanation (pos+neg), pred {pred_top1_name}, GT {label_name}'
                    lime_kwargs = {'positive_only': False, 'num_features': 10, 'hide_rest': False}
                    output_path = join(lime_dir, f'{idx[i]}_pos_neg.png')

                    show_lime_image(explanation, img_for_lime, lime_kwargs, title, output_path=output_path)
                    # show_lime_image(explanation, img_for_lime, lime_kwargs, title, show=args.show)

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
                             labels=label_names, show=args.show)

            cm_binary_location = join(output_dir, 'confusion_matrix_binary.pdf')
            label_names_binary = ['genuine', 'attack']
            preds_binary = outputs_test['preds'] != bona_fide
            labels_binary = outputs_test['labels'] != bona_fide

            confusion_matrix(labels_binary, preds_binary, output_location=cm_binary_location, labels=label_names_binary,
                             show=args.show)

    ''' CAM Explanations  '''
    if args.cam:
        """
        Features:
        - AUC
            - per-class  #done#
            - how to normalize
        Baselines:
        - blur image instead of black  #done#
        - random weights
        - sobel explanation
        - centered circle explanation
        
        - check that percent_kept = 0 leads to black image  #done#
        """
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            LayerCAM
        # todo check more methods in the library
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
        from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
        import cv2
        from src.util_image import deprocess

        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)
        target_layers = [model.features[-1][0]]  # [model.layer4[-1]]  # resnet18
        # ^ make sure only last layer of the block is used, but still wrapped in a list
        method_modules = [GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM,
                          EigenCAM]  # ScoreCAM OOM, FullGrad too different; AblationCAM runs long
        # methods_callables = [ for method in methods_]
        # grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputSoftmaxTarget(cat) for cat in range(config_dict['num_classes'])]
        #           ^ minor visual difference from ClassifierOutputTarget.
        cam_metric = CamMultImageConfidenceChange()  # DropInConfidence()
        percentages_kept = [100, 90, 70, 50, 30, 10, 0]
        perturbed_percentages_kept = percentages_kept[1:]  # 100% was the original, so skip it

        cams_out = []  # list of dicts: { idx: int, label: int, pred: int, path: str, cam: np.array[C, H, W], TODO finish }
        for i_m, method_module in enumerate(method_modules):
            cam_method = method_module(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')
            method_name = cam_method.__class__.__name__
            print(f'{method_name} ({i_m + 1}/{len(method_modules)})')
            if 'batch_size' in cam_method.__dict__:  # AblationCAM
                print(f'Orig batch size for {method_name}: {cam_method.batch_size}')
                cam_method.batch_size = 128  # default == 32; adan (8gb) can handle 128

            for batch in tqdm(test_loader, mininterval=2., desc=method_name):
                ''' Predict (batch) '''
                img_batch = batch['image'].to(device)
                preds_b, preds_classes_b = predict(model, img_batch)  # prediction on original image (batch)

                ''' Generate CAMs for images in batch '''
                for i, img in enumerate(img_batch):
                    img_np = img.cpu().numpy()
                    img_plotting = deprocess(img_np)
                    pred = preds_b[i]
                    pred_class = preds_classes_b[i]
                    pred_class_name = label_names[pred_class]
                    idx = batch['idx'][i].item()
                    label = batch['label'][i].item()

                    ''' Generate CAMs for all classes (targets) '''
                    cams = []
                    for j, t in enumerate(targets):
                        grayscale_cam = cam_method(input_tensor=img[None, ...], targets=[t])  # img 4D
                        cams.append(grayscale_cam)

                    # only work with CAM for the predicted class
                    cam_pred = cams[pred_class][0]

                    # resize to original image size
                    wh_image = (img.shape[1], img.shape[2])
                    if cam_pred.shape != wh_image:
                        cam_pred = cv2.resize(cam_pred, wh_image)

                    ''' Perturbation baselines '''
                    b_black = np.zeros_like(cam_pred)
                    b_blurred = cv2.blur(img_np, (config.blur_img_s, config.blur_img_s))
                    b_blurred_c = np.stack(
                        [cv2.blur(img_np[c], (config.blur_img_s, config.blur_img_s)) for c in range(3)])
                    b_mean = np.zeros_like(img_np) + np.mean(img_np, axis=(1, 2), keepdims=True)
                    b_blurcdark = (b_black + b_blurred_c) / 2
                    # b_blurred_hwc = b_blurred.transpose(1, 2, 0)
                    # b_blurred_hwc_mean = b_blurred_hwc - np.mean(b_blurred_hwc, axis=(0, 1), keepdims=True)
                    # plot_many(img_plotting, b_black, b_blurred_c, b_mean, b_blurcdark)
                    # TODO Baseline Sobel
                    # TODO Baseline Centered Circle
                    # TODO Baseline

                    ''' Perturb explained region '''
                    cam_blurred = cam_pred + cv2.blur(cam_pred, (config.blur_cam_s, config.blur_cam_s))
                    thresholds = np.percentile(cam_blurred, perturbed_percentages_kept)
                    thresholds[-1] = cam_blurred.min()  # make sure 0% kept is 0

                    for base_name, baseline in zip(['black', 'blur', 'mean', 'blurdark'],
                                                   [b_black, b_blurred, b_mean, b_blurcdark]):  # TODO try
                        baseline = torch.tensor(baseline, device=device)
                        scores_perturbed = [pred[pred_class]]  # [score of 100% kept (original)], skip in loop
                        imgs_perturbed = []
                        for th, pct in zip(thresholds, perturbed_percentages_kept):
                            mask = torch.Tensor(cam_blurred < th).to(device)
                            img_masked = (img * mask + (1 - mask) * baseline)[None, ...]
                            # img_masked_plotting = img_plotting * mask.cpu().numpy()
                            imgs_perturbed.append(img_masked)

                        ''' Predict on perturbed image '''
                        preds_perturbed_b, _ = predict(model, torch.cat(imgs_perturbed, dim=0))

                        ''' Compute Deletion Score '''
                        scores_perturbed = [pred[pred_class]] + list(preds_perturbed_b[:, pred_class])

                        cams = np.stack(cams)  # [C, H, W]
                        cams = (255 * cams).astype(np.uint8)  # uint8 to save space

                        cams_out.append({'cam': cams, 'idx': idx,
                                         'method': method_name, 'path': batch['path'][i],  # strings are not tensors
                                         'baseline': base_name,
                                         'label': label, 'pred': preds_classes_b[i],
                                         'del_scores': scores_perturbed, 'pred_scores': preds_b[i],
                                         'percentages_kept': percentages_kept})
                    # end of batch
                # end of dataset
            # end of method

            # free gpu memory
            # del cam_method
            # torch.cuda.empty_cache()
            ''' Save CAMs per method '''
            cams_df = pd.DataFrame(cams_out)
            output_path = join(cam_dir, f'cams_{method_name}_{limit if limit else "full"}.pkl.gz')
            cams_df.to_pickle(output_path, compression='gzip')
            # print('Not saving CAMs!')
        # end of all methods

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
    print(f'Execution finished in {t1 - t0:.2f}s')  # since dataset loaded

    ''' Sandbox Area'''
    if args.cam:
        # plot deletion scores per-baseline
        baselines = cams_df.baseline.unique()
        xs = cams_df.percentages_kept.iloc[0]

        fig, ax = plt.subplots(len(baselines), 1, figsize=(6, 10), sharex=True)
        for k, base_name in enumerate(baselines):
            ax[k].set_title(base_name)
            cams_df_base = cams_df[cams_df.baseline == base_name]
            del_scores = np.stack(cams_df_base.del_scores.to_numpy())
            for i, c in enumerate(label_names):
                idxs = cams_df_base.label == i
                dsi = del_scores[idxs]
                ax[k].plot(xs, dsi.mean(axis=0), label=c)

        plt.ylabel('Score')
        plt.xlabel('% Pixels Kept')
        plt.suptitle('Deletion Scores per Baseline')
        plt.gca().invert_xaxis()  # reverse x-axis
        # add minor y ticks

        # figure legend out right top
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    if args.cam:
        pass
