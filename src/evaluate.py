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

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use('tkagg')  # helped for some plotting issues (console run, non-pycharm)
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
    - preprocess is obtained automatically in load_model, and applied in DataLoader. Do not use it manually.
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
    print(f'Running on device: {device}')
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    np.set_printoptions(precision=3, suppress=True)  # human-readable printing

    seed = args.seed if args.seed is not None else config.seed_eval_default  # 42
    print(f'Random seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # todo extract init and seed application to respective functions [clean]

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

    ''' LIME Explanations '''
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

    ''' CAM Explanations  '''
    if args.cam:
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            LayerCAM
        # todo check more methods in the library
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
        from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
        from src.util import get_marker
        import cv2


        def deprocess(img):
            import torch
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()  # might fail when no grad
                img = img.transpose(1, 2, 0)
            img = img - np.mean(img)
            img = img / (np.std(img) + 1e-5)
            img = img * 0.1
            img = img + 0.5
            img = np.clip(img, 0, 1)
            # don't make image uint8
            return img


        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)
        target_layers = [model.features[-1][0]]  # [model.layer4[-1]]  # resnet18
        # ^ make sure only last layer of the block is used, but still wrapped in a list
        method_modules = [
            GradCAM]  # [GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM]  # ScoreCAM OOM, FullGrad too different
        # methods_callables = [ for method in methods_]
        # grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputSoftmaxTarget(cat) for cat in range(config_dict['num_classes'])]
        #           ^ minor visual difference from ClassifierOutputTarget.
        cam_metric = CamMultImageConfidenceChange()  # DropInConfidence()
        percentages_kept = [100, 90, 70, 50, 30, 10, 0]

        cams_out = []  # list of dicts: { idx: int, label: int, pred: int, path: str, cam: np.array[C, H, W], TODO finish }
        for method_module in method_modules:
            cam_method = method_module(model=model, target_layers=target_layers, use_cuda=True)
            method_name = cam_method.__class__.__name__
            print(f'{method_name}:')
            if 'batch_size' in cam_method.__dict__:  # AblationCAM
                print(f'Orig batch size for {method_name}: {cam_method.batch_size}')
                cam_method.batch_size = 32  # == default

            for batch in tqdm(test_loader, mininterval=2., desc=method_name):
                ''' Predict (batch) '''
                img_batch = batch['image'].to(device)
                with torch.no_grad():  # prediction on original image (batch)
                    preds_raw_b = model(img_batch).cpu()
                    preds_b = F.softmax(preds_raw_b, dim=1).numpy()
                    preds_classes_b = np.argmax(preds_b, axis=1)

                ''' Generate CAMs for images in batch '''
                for i, img in enumerate(img_batch):
                    img_plotting = deprocess(img.cpu().numpy())
                    pred = preds_b[i]
                    pred_class = preds_classes_b[i]
                    pred_class_name = label_names[pred_class]
                    idx = batch['idx'][i].item()
                    label = batch['label'][i].item()
                    cams = []

                    ''' Generate CAMs for all classes (targets) '''
                    for j, t in enumerate(targets):
                        grayscale_cam = cam_method(input_tensor=img[None, ...], targets=[t])  # img 4D
                        cams.append(grayscale_cam)

                    # only work with CAM for the predicted class
                    cam_pred = cams[pred_class][0]

                    # resize to original image size
                    wh_image = (img.shape[1], img.shape[2])
                    if cam_pred.shape != wh_image:
                        cam_pred = cv2.resize(cam_pred, wh_image)

                    ''' Remove explained region from image '''
                    cam_blurred = cam_pred + cv2.blur(cam_pred, (config.blur_kernel_size, config.blur_kernel_size))
                    cams_perturbed = [cam_blurred]
                    scores_perturbed = [pred[pred_class]]  # [score of 100% kept (original)], skip in loop
                    for expl_percent_kept in percentages_kept[1:]:
                        threshold = np.percentile(cam_blurred, expl_percent_kept)
                        cam_pred_mask = torch.Tensor(cam_blurred <= threshold).cuda()
                        cams_perturbed.append(cam_pred_mask)

                        ''' Remove explained region '''
                        img_masked = (img * cam_pred_mask)[None, ...]
                        img_masked_plotting = img_plotting * cam_pred_mask.cpu().numpy()
                        # print(f'explanation kept: {expl_percent_kept} %')
                        total_percent_kept = 100 * cam_pred_mask.sum() / cam_pred_mask.numel()
                        # print(f'total image kept: {total_percent_kept:.2f} %')  # now should be same as expl_percent_kept

                        ''' Predict on perturbed image '''
                        with torch.no_grad():  # prediction on modified image (single, not batched)
                            preds_raw_perturbed_b = model(img_masked).cpu()
                            preds_perturbed_b = F.softmax(preds_raw_perturbed_b, dim=1).numpy()
                            preds_classes_perturbed_b = np.argmax(preds_perturbed_b, axis=1)
                            pred_score_perturbed = preds_perturbed_b[0, preds_classes_b][0]
                            # not taking the top class for perturbed prediction ^^^.

                        ''' Compute Deletion Score '''
                        deletion_score = pred_score_perturbed - pred[pred_class]
                        # print(f'orig: {pred}\npert: {preds_perturbed_b}\nscore: {pred_score_perturbed:.4f}, drop: {deletion_score:.4f}\n')
                        scores_perturbed.append(pred_score_perturbed)

                        # cam2 = cam_method(input_tensor=img_masked, targets=[targets[pred_class]])
                        output_path = join(cam_dir, f'deletion_{idx}_{method_name}_kept{expl_percent_kept}.png')
                        plot_many(img_plotting, cam_pred, cam_pred_mask, img_masked_plotting,
                                  title=f'{method_name}, idx: {idx}\npredicted: {pred_class_name} ({pred[pred_class]:.4f})\n'
                                        f'{expl_percent_kept}% kept, drop: {deletion_score:.4f}',
                                  titles=['original', 'cam', 'mask', 'masked'], output_path=output_path, show=False)

                    cams = np.stack(cams)  # [C, H, W]
                    cams = (255 * cams).astype(np.uint8)  # uint8 to save space

                    cams_out.append({'cam': cams, 'idx': idx,
                                     'method': method_name, 'path': batch['path'][i],  # strings are not tensors
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
            cams_df.to_pickle(join(cam_dir, f'cams-{method_name}.pkl.gz'), compression='gzip')
            # print('Not saving CAMs!')
        # end of all methods

        ''' Plot Deletion Scores '''
        if False:
            scores = np.stack(cams_df['del_scores'].values)  # todo stack not necessary [clean]
            idxs = cams_df['idx'].values
            # scores - axis 0 is perturbation intensity, axis 1 is sample
            # x = perturbation_intensities
            # y = scores
            # hue = sample
            import seaborn as sns

            plt.figure(figsize=(6, 4))
            for i, scores_line in enumerate(scores):
                sample_idx = idxs[i]
                marker_random = get_marker(sample_idx)
                plt.plot(percentages_kept, scores_line, label=sample_idx, marker=marker_random)

            plt.xticks(percentages_kept)  # original x-values ticks
            plt.gca().invert_xaxis()  # decreasing x axis
            plt.ylim(0, 1.05)
            plt.legend(title='Sample ID')
            plt.ylabel('Prediction Score')
            plt.xlabel('Perturbation Intensity (% of image kept)')
            plt.title('Deletion Metric')

            # remove top and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.tight_layout()
            output_path = join(cam_dir, 'deletion_metric-samples.png')
            if output_path:
                plt.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
            if args.show:
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
