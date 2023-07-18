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
from metrics import compute_metrics, confusion_matrix
from util import print_dict, save_i, keys_append
from util_torch import init_device, init_seed, load_model_eval, get_dataset_module, predict, eval_loop
from util_image import plot_many
from dataset_base import pick_dataset_version, load_dataset
import config

''' Global variables '''
run_dir = ''
model = None
device = None
preprocess = None
criterion = None

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


def eval_loop_wrapper(loader):
    global model, criterion, device
    paths, labels, preds, loss = eval_loop(model, loader, criterion, device)
    metrics = compute_metrics(labels, preds)
    # log results
    res_epoch = {'Loss': loss,
                 'Accuracy': metrics['Accuracy']}
    predictions = pd.DataFrame({'path': paths, 'label': labels, 'pred': preds})
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
    paths_save = join(output_dir, f'predictions_{split_name}.pkl')
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


# def main():
#     global model, device, criterion  # disable when not using def main
if __name__ == '__main__':
    """
    Evaluate model on dataset, run explanation methods
    Note:
    - preprocess is obtained automatically in load_model, and applied in DataLoader. Do not use it manually.
    """
    ''' Parse arguments '''
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    if sum([args.lime, args.cam, args.eval, args.emb]) > 1:
        raise ValueError('Choose only one of: lime/cam/eval/emb')

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
    dataset_meta = pick_dataset_version(args.dataset, training_mode, attack=args.attack)
    attack_test = args.attack if args.attack else dataset_meta['attack_test']  # unseen/one-attack
    dataset_id = f'{args.dataset}-{training_mode}' + f'-{attack_test}' if attack_test else ''
    loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                     'seed': seed, 'drop_last': False,
                     'transform_train': preprocess['eval'], 'transform_eval': preprocess['eval']}
    #                                               ^^^^ note: eval transform is used for both train and test
    train_loader, val_loader, test_loader = \
        load_dataset(dataset_meta, dataset_module, path_prefix=args.path, limit=limit, quiet=False, **loader_kwargs)
    bona_fide = dataset_module.bona_fide
    label_names = dataset_module.label_names_unified
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
        ''' Training set '''
        outputs_train = get_preds(train_loader, 'train', output_dir)
        ''' Validation set '''
        outputs_val = get_preds(val_loader, 'val', output_dir, new=True)
        ''' Test set '''
        outputs_test = get_preds(test_loader, 'test', output_dir)
        metrics_test = compute_metrics(outputs_test['label'], outputs_test['pred'])
        metrics_test = keys_append(metrics_test, ' Test')
        cr = classification_report(outputs_test['label'], outputs_test['pred'])
        print(cr)

        ''' Confusion matrix '''
        if True:
            cm_location = join(output_dir, f'confmat_test.pdf')
            confusion_matrix(outputs_test['label'], outputs_test['pred'], output_location=cm_location,
                             labels=label_names, show=args.show)

            cm_binary_location = join(output_dir, f'confmat_binary_test.pdf')
            label_names_binary = ['genuine', 'attack']
            preds_binary = outputs_test['pred'] != bona_fide
            labels_binary = outputs_test['label'] != bona_fide
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
        - random weights
        - sobel explanation  #done#
        - centered circle explanation  #done#
        
        - check that percent_kept = 0 leads to black image  #done#
        
        format:
         list of dicts: { idx: int, label: int, pred: int, path: str, cam: np.array[C, H, W], TODO finish }
        """
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            LayerCAM  # todo check more methods in the library
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
        import cv2
        from util_image import SobelFakeCAM, CircleFakeCAM, deprocess

        cam_dir = join(output_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)
        target_layers = [model.features[-1][0]] if model_name == 'efficientnet_v2_s' else [model.layer4[-1].bn2]
        # ^ make sure only last layer of the block is used, but still wrapped in a list
        method_modules = [HiResCAM] + [SobelFakeCAM, CircleFakeCAM, GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM,
                                       EigenCAM]  # ScoreCAM OOM, FullGrad too different; AblationCAM runs long
        targets = [ClassifierOutputSoftmaxTarget(cat) for cat in range(config_dict['num_classes'])]
        #           ^ minor visual difference from ClassifierOutputTarget.
        # cam_metric = CamMultImageConfidenceChange()  # DropInConfidence()

        perturbed_percentages_kept = config.percentages_kept[1:]  # 100% was the original, so skip it

        for i_m, method_module in enumerate(method_modules):
            cams_out = []
            cam_method = method_module(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')
            method_name = cam_method.__class__.__name__
            print(f'{method_name} ({i_m + 1}/{len(method_modules)})')
            if 'batch_size' in cam_method.__dict__:  # AblationCAM
                print(f'Orig batch size for {method_name}: {cam_method.batch_size}')
                cam_method.batch_size = 128  # default == 32; adan (8gb) can handle 128

            # if False:
            #     batch = next(iter(test_loader))
            #     i = 1
            #     img = img_batch[i]

            for batch in tqdm(test_loader, mininterval=2., desc=method_name):
                ''' Predict (batch) '''
                img_batch = batch['image'].to(device)
                preds_b, preds_classes_b = predict(model, img_batch)  # prediction on original image (batch)

                ''' Generate CAMs for images in batch '''
                for i, img in enumerate(img_batch):
                    img_np = img.cpu().numpy()
                    # if 'image_orig' in batch:
                    #     img_plotting = batch['image_orig'][i].cpu().numpy().transpose(1, 2, 0)
                    # else:
                    #     img_plotting = deprocess(img_np)
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
                        cam_pred = cv2.resize(cam_pred, wh_image)  # interpolation bilinear by default

                    ''' Perturbation baselines '''
                    b_black = np.zeros_like(cam_pred)
                    # b_blur = np.stack(
                    #     [cv2.blur(img_np[c], (config.blur_img_s, config.blur_img_s)) for c in range(3)])
                    # b_mean = np.zeros_like(img_np) + np.mean(img_np, axis=(1, 2), keepdims=True)
                    # b_blurdark = (b_black + b_blur) / 4
                    # # plot_many(img_plotting, b_black, b_blur, b_mean, b_blurdark)

                    ''' Perturb explained region '''
                    cam_blurred = cam_pred + 1 / 10 * cv2.blur(cam_pred, (config.blur_cam_s, config.blur_cam_s))
                    thresholds = np.percentile(cam_blurred, perturbed_percentages_kept)
                    thresholds[-1] = cam_blurred.min()  # make sure 0% kept is 0

                    # for base_name, baseline in zip(['black', 'blur', 'mean', 'blurdark'],
                    #                                [b_black, b_blur, b_mean, b_blurdark]):  # unused for main experiments
                    for base_name, baseline in [('black', b_black)]:
                        baseline = torch.tensor(baseline, device=device)
                        scores_perturbed = [pred[pred_class]]  # [score of 100% kept (original)], skip in loop
                        imgs_perturbed = []
                        for th, pct in zip(thresholds, perturbed_percentages_kept):
                            mask = torch.Tensor(cam_blurred < th).to(device)
                            img_masked = (img * mask + (1 - mask) * baseline)[None, ...]
                            # img_masked_plotting = img_plotting * mask.cpu().numpy()
                            imgs_perturbed.append(img_masked.to(dtype=torch.float32))

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
                                         'percentages_kept': config.percentages_kept})
                    # end of batch
                # end of dataset
            # end of method

            # free gpu memory
            # del cam_method
            # torch.cuda.empty_cache()
            ''' Save CAMs per method '''
            cams_df = pd.DataFrame(cams_out)
            output_path = join(cam_dir, f'cams_{method_name}_{limit if limit else "full"}.pkl.gz')
            if exists(output_path):  # rename old file
                print(f'File {output_path} already exists, renaming it.')
                # prepend 'old' to file name
                move(output_path, output_path.replace('cams_', 'old-cams_'))
            cams_df.to_pickle(output_path)
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

        ''' Eval loop '''
        len_loader = len(test_loader)
        ep_loss = 0.0
        avgpools = []
        preds = []
        labels = []
        paths = []
        features = []
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

    ''' Sandbox Area'''
    if False and args.cam:
        cams_df = None
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
