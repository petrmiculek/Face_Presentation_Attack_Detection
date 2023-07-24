# stdlib
import argparse
import logging
import os
import json
import sys
import time
from os.path import join, exists
from shutil import move

# fix for local import problems
sys.path.append(os.getcwd())  # + [d for d in os.listdir() if os.path.isdir(d)]

# external
from sklearn.metrics import classification_report
from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

import pandas as pd
import torch
import numpy as np

# matplotlib.use('tkagg')  # for plotting issues (local console run, not on metacentrum)
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
from metrics import compute_eer, compute_metrics, plot_confusion_matrix, plot_roc_curve
from util import print_dict, save_i, keys_append
from util_torch import init_device, init_seed, load_model_eval, get_dataset_module, predict, eval_loop
from explanations import perturbation_baselines, perturbation_masks
from dataset_base import pick_dataset_version, load_dataset, get_dataset_setup, label_names_binary
import config

''' Global variables '''
run_dir = ''
model = None
device = None
preprocess = None
criterion = None
args = None
bona_fide = None
label_names = None

''' Parsing Arguments '''
parser = argparse.ArgumentParser()  # description='Evaluate model on dataset, run explanation methods'
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=None)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', default=None)
parser.add_argument('-k', '--attack', help='test attack for unseen_attack/one_attack (1..C)', type=int, default=None)
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=None)
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
# explain interplay of dataset and path
parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, required=True)  # rose_youtu-single
parser.add_argument('-p', '--path', help='path to dataset samples', type=str, required=True)
parser.add_argument('-r', '--run', help='model/dataset/settings to load (run directory)', default=None)
parser.add_argument('-l', '--lime', help='generate LIME outputs', action='store_true')
parser.add_argument('-c', '--cam', help='generate CAM outputs', action='store_true')
parser.add_argument('-v', '--eval', help='run evaluation loop', action='store_true')
parser.add_argument('-e', '--emb', help='get embeddings', action='store_true')
parser.add_argument('-z', '--show', help='show outputs', action='store_true')


# possibly add --no-log


def eval_loop_wrapper(loader):
    """ Run evaluation loop and log results. """
    global model, criterion, device
    paths, labels, preds, probs, loss = eval_loop(model, loader, criterion, device)
    metrics = compute_metrics(labels, preds)
    # log results
    res_epoch = {'Loss': loss,
                 'Accuracy': metrics['Accuracy']}
    predictions = pd.DataFrame({'path': paths, 'label': labels, 'pred': preds, 'prob': list(probs)})
    return res_epoch, predictions


def get_preds(dataset_loader, split_name, output_dir, new=False, overwrite=False):
    """
    Get predictions for a dataset split -- load from file or run eval loop.
    :param dataset_loader: dataset loader
    :param split_name: train/val/test
    :param output_dir: directory to save predictions
    :param new: force new predictions
    :param overwrite: overwrite when saving
    :return: predictions dataframe
    """
    limit = 'full' if args.limit in [None, -1] else args.limit
    paths_save = join(output_dir, f'predictions_{split_name}_{limit}.pkl')
    if new or not os.path.isfile(paths_save):
        print(f'Evaluating on: {split_name}')
        res, predictions = eval_loop_wrapper(dataset_loader)
        print_dict(res)
        save_i(paths_save, predictions, overwrite=overwrite)
        predictions.to_pickle(paths_save, )
    else:
        # load predictions from file
        print(f'Loading {split_name} set results from file')
        predictions = pd.read_pickle(paths_save)

    return predictions


def plot_confmats(df_outputs, output_dir, split='test'):
    """ Plot confusion matrices and ROC curve comfortably. """
    global label_names, bona_fide, args
    ''' Multi-class confusion matrix '''
    output_path = join(output_dir, f'confmat_{split}.pdf')
    plot_confusion_matrix(df_outputs['label'], df_outputs['pred'], output_path=output_path,
                          labels=label_names, show=args.show)
    ''' Binary confusion matrix '''
    output_path = join(output_dir, f'confmat_binary_{split}.pdf')
    preds_binary = df_outputs['pred'] != bona_fide
    labels_binary = df_outputs['label'] != bona_fide
    plot_confusion_matrix(labels_binary, preds_binary, output_path=output_path,
                          labels=label_names_binary, show=args.show)


def roc_eer(df_outputs, output_dir, split='test'):
    """ Plot ROC curve and compute Equal Error Rate (EER). """
    global bona_fide, args
    ''' ROC-Curve '''
    probs = 1 - np.stack(df_outputs['prob'].values)[:, 0]  # negative bona_fide probability
    gts_binary = np.float32(df_outputs['label'] != bona_fide)
    plot_roc_curve(gts_binary, probs, show=args.show, output_path=join(output_dir, f'roc_{split}.pdf'))
    ''' Equal Error Rate (EER) '''
    eer, eer_threshold = compute_eer(gts_binary, probs)
    print(f'EER: {eer:.2f}%, threshold: {eer_threshold:.2f}')
    return eer, eer_threshold


# def main():
#     global model, device, criterion, args, bona_fide, label_names, limit  # disable when not using def main
if __name__ == '__main__':
    """
    Evaluate model on dataset, run explanation methods
    Note:
    - preprocess is obtained automatically in load_model, and applied in DataLoader. Do not use it manually.
    """
    ''' Parse arguments '''
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    run_dir = args.run
    ''' Load run setup/configuration '''
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    print('Loading model and setup from:', run_dir)
    ''' Arguments '''
    batch_size = args.batch_size if args.batch_size else config_dict['batch_size']
    num_workers = args.num_workers if args.num_workers is not None else config_dict['num_workers']
    limit = args.limit if args.limit else -1
    training_mode = args.mode if args.mode else config_dict['mode']  # 'all_attacks'
    ''' Initialization '''
    device = init_device()
    seed = args.seed if args.seed is not None else config.seed_eval_default
    init_seed(seed)
    ''' Model '''
    model_name = config_dict['model_name'] if 'model_name' in config_dict else config_dict['arch']
    model, preprocess = load_model_eval(model_name, config_dict['num_classes'], run_dir, device)
    print(f'Model: {model_name} with {config_dict["num_classes"]} classes')
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss
    ''' Dataset '''
    dataset_module = get_dataset_module(args.dataset)
    attack = args.attack if args.attack else config_dict['attack']
    dataset_meta = pick_dataset_version(args.dataset, training_mode, attack=attack)
    attack_test = args.attack if args.attack else dataset_meta['attack_test']  # unseen/one-attack
    dataset_id = f'{args.dataset}-{training_mode}' + f'-{attack_test}' if attack_test else ''
    loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                     'seed': seed, 'drop_last': False,
                     'transform_train': preprocess['eval'], 'transform_eval': preprocess['eval']}
    #                                               ^^^^ note: eval transform is used for both train and test
    train_loader, val_loader, test_loader = \
        load_dataset(dataset_meta, dataset_module, path_prefix=args.path, limit=limit, quiet=False, **loader_kwargs)
    bona_fide = dataset_module.bona_fide
    label_names, num_classes = get_dataset_setup(dataset_module, training_mode)

    ''' Output directory '''
    output_dir = join(run_dir, f'eval-{dataset_id}')
    os.makedirs(output_dir, exist_ok=True)
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
        preds = np.array(preds)  # todo maybe concatenate instead?
        idxs = np.array(idxs)

        np.savez(join(lime_dir, 'lime.npz'), labels=labels, paths=paths, preds=preds, idxs=idxs)  # todo pandas
        print(f'LIME explanations saved to {lime_dir}')

    ''' Evaluation '''
    if args.eval:
        ''' Evaluate model on training/validation/test set, save predictions '''
        outputs_train = get_preds(train_loader, 'train', output_dir)
        outputs_val = get_preds(val_loader, 'val', output_dir, new=True)
        outputs_test = get_preds(test_loader, 'test', output_dir)
        # note: no notion of `limit` when saving predictions. Assumed to be the full dataset length.
        metrics_test = compute_metrics(outputs_test['label'], outputs_test['pred'])
        metrics_test = keys_append(metrics_test, ' Test')
        cr = classification_report(outputs_test['label'], outputs_test['pred'])
        print(cr)
        ''' Confusion matrix + ROC curve '''
        plot_confmats(outputs_test, output_dir, 'test')
        eer_train, eer_th_train = roc_eer(outputs_train, output_dir, 'train')
        eer_val, eer_th_val = roc_eer(outputs_val, output_dir, 'val')
        eer_test, eer_th_test = roc_eer(outputs_test, output_dir, 'test')

    ''' CAM Explanations  '''
    if args.cam:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
        import cv2

        cam_dir = join(output_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)
        ''' Setup: model, target layers and classes, methods '''
        target_layers = [model.features[-1][0]] if model_name == 'efficientnet_v2_s' else [model.layer4[-1].bn2]
        # ^ make sure only last layer of the block is used, but still wrapped in a list
        targets = [ClassifierOutputSoftmaxTarget(cat) for cat in range(num_classes)]
        #           ^ minor visual difference from ClassifierOutputTarget.
        percentages = config.percentages_kept  # %pixels kept when perturbing the images
        len_pct = len(percentages)
        method_modules = [GradCAM]  # [GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, RandomCAM, SobelFakeCAM, CircleFakeCAM]
        # ScoreCAM OOM, FullGrad too different; AblationCAM runs too long
        ''' Generate CAMs using all methods '''
        for i_m, method_module in enumerate(method_modules):
            cams_out = []
            cam_method = method_module(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')
            method_name = cam_method.__class__.__name__
            print(f'{method_name} ({i_m + 1}/{len(method_modules)})')
            progress_bar = tqdm(test_loader, mininterval=2., desc=method_name)
            for batch in progress_bar:
                ''' Predict on original images '''
                preds_b, preds_classes_b = predict(model, batch['image'].to(device))  # prediction on original image (batch)
                for i, img in enumerate(batch['image']):
                    progress_bar.set_description(f'{method_name}:{i}/{batch_size}')
                    img_np = img.cpu().numpy()
                    pred = preds_b[i]
                    pred_class = preds_classes_b[i]
                    idx = batch['idx'][i].item()
                    label = batch['label'][i].item()
                    ''' Generate CAMs for all classes (targets) '''
                    cams = [cam_method(input_tensor=img[None, ...], targets=[t]) for t in targets]
                    cam_pred = cams[pred_class][0]  # deletion metric only works with CAM for the predicted class
                    cams = np.stack(cams)  # [C, H, W]
                    cams = (255 * cams).astype(np.uint8)[:, 0]  # uint8 to save space
                    # resize CAM to original image size
                    cam_pred = cv2.resize(cam_pred, (img.shape[1], img.shape[2]))  # interpolation bilinear by default
                    ''' Perturb explained region '''
                    masks = perturbation_masks(cam_pred, percentages)
                    baselines = perturbation_baselines(img_np, which=['black', ])  #
                    for base_name, baseline in baselines.items():
                        ''' Predict on perturbed images - deletion and insertion metrics '''
                        imgs_deletion = np.array([(img_np * mask + (1 - mask) * baseline) for mask in masks])
                        imgs_insertion = np.array([(img_np * (1 - mask) + mask * baseline) for mask in masks])
                        preds_both, _ = predict(model, torch.tensor(np.r_[imgs_deletion, imgs_insertion], device=device, dtype=torch.float32))
                        probs_deletion, probs_insertion = preds_both[:len_pct], preds_both[len_pct:][::-1]
                        # `probs_insertion` reversed to match decreasing order of percentages_kept    ^^^^
                        ''' Save Perturbation Scores '''
                        cams_out.append({'cam': cams, 'idx': idx, 'path': batch['path'][i],  # strings are not tensors
                                         'method': method_name, 'percentages_kept': config.percentages_kept,
                                         'baseline': base_name, 'label': label, 'pred': preds_classes_b[i], 'pred_scores': pred,
                                         'del_scores': probs_deletion, 'ins_scores': probs_insertion})
                    # end of batch
                # end of dataset
            # end of method
            ''' Save CAMs per method '''
            cams_df = pd.DataFrame(cams_out)
            output_path = join(cam_dir, f'cams_{method_name}_{limit if limit else "full"}.pkl.gz')
            if exists(output_path):  # rename: prepend 'old' to file name
                print(f'File {output_path} already exists, renaming it.')
                move(output_path, output_path.replace('cams_', 'old-cams_'))
            cams_df.to_pickle(output_path)
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
        #     raise AttributeError('Model does not have fw_with_emb method')
        ''' Extract embeddings through forward hooks '''
        activation = {}


        def get_activation(name):
            """ Hook for extracting activations from a layer specified by name """

            def hook(model, inputs, outputs):
                activation[name] = outputs.detach()  # refers to outer-scope `activation` dict

            return hook


        emb_layer = 'avgpool'  # resnet18
        # emb_layer = 'avgpool'  # efficientnet_v2_s
        embeddings_hook = model.avgpool.register_forward_hook(get_activation(emb_layer))

        ''' Embeddings Eval loop '''
        len_loader = len(test_loader)
        ep_loss = 0.0
        preds, labels, paths, features, avgpools = [], [], [], [], []
        # todo: to eval_loop - add fw_hook, save avgpools [clean]
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
