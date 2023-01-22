# stdlib
import argparse
import json
import logging
import os
import datetime
from os.path import join
from copy import deepcopy

"""
# fix for local import problems - add all local directories
import sys
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)
"""

# external
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

os.environ["WANDB_SILENT"] = "true"

import wandb as wb

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
import dataset_rose_youtu as dataset
from metrics import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, xor, keys_append, save_dict_json
from model_util import EarlyStopping
import resnet18

# run_dir = 'runs/2023-01-10_14-41-03'  # 'unseen_attack'
run_dir = 'runs/2023-01-10_15-12-22'  # 'all_attacks'

model = None
device = None
preprocess = None
criterion = None

''' Global variables '''
# -

''' Parsing Arguments '''
parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
# parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
# parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)',
#                     type=str, default=None)
# parser.add_argument('-d', '--dataset', help='dataset to evaluate on', type=str, default=None)
parser.add_argument('-r', '--run', help='model/dataset/settings to load', type=str, default=None)


def eval_loop(loader):
    len_loader = len(loader)
    ep_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for img, label in tqdm(loader, leave=False, mininterval=1., desc='Eval'):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, label)
            ep_loss += loss.item()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)

            # save prediction
            labels.append(label.cpu().numpy())
            preds.append(prediction_hard.cpu().numpy())

    # loss is averaged over batch already, divide by batch number
    ep_loss /= len_loader

    metrics = compute_metrics(labels, preds)

    # log results
    res_epoch = {
        'Loss': ep_loss,
        'Accuracy': metrics['Accuracy'],
    }

    return res_epoch, preds, labels


# def main():
#     global model, device, preprocess, criterion  # disable when not using def main
if __name__ == '__main__':
    args = parser.parse_args()
    run_dir = args.run
    # read setup from run folder
    with open(join(run_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    print('Loading model and setup from:', run_dir)

    # todo: in eval allow to overwrite config with args
    # if args.batch_size is not None:
    #     config_dict['batch_size'] = args.batch_size
    # if args.num_workers is not None:
    #     config_dict['num_workers'] = args.num_workers
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

    ''' Load Model '''
    model_name = config_dict['model_name']
    # def load_model(model_name, weights, weight_class, num_classes): ...
    if model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18.resnet18(weights=weights, weight_class=ResNet18_Weights)
        model.fc = torch.nn.Linear(512, config_dict['num_classes'], bias=True)
        preprocess = weights.transforms()
    elif model_name == 'efficientnet_v2_s':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights,
                                  weight_class=EfficientNet_V2_S_Weights)  # todo possibly set num_classes=2
        model_name = efficientnet_v2_s.__name__
        preprocess = weights.transforms()
    else:
        raise ValueError(f'Unknown model name {model_name}')

    model.load_state_dict(torch.load(join(run_dir, 'model_checkpoint.pt')))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss

    ''' Load Data '''
    if True:
        from dataset_base import pick_dataset_version, load_dataset

        dataset_meta = pick_dataset_version(dataset_name, training_mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': config_dict['batch_size'],
                         'num_workers': config_dict['num_workers'], 'pin_memory': True}
        train_loader, val_loader, test_loader = \
            load_dataset(dataset_meta, dataset_module, limit=-1, quiet=False, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

    if False:
        ''' Evaluation '''
        print('Training set')
        res_train, preds_train, labels_train = eval_loop(train_loader)
        print_dict(res_train)

        print('Validation set')
        res_val, preds_val, labels_val = eval_loop(val_loader)
        print_dict(res_val)

        print('Test set')
        res_test, preds_test, labels_test = eval_loop(test_loader)
        print_dict(res_test)

    ''' Explainability '''

    if False:
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
            FullGrad
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        cam_dir = join(run_dir, 'cam')
        os.makedirs(cam_dir, exist_ok=True)

        target_layers = [model.layer4[-1]]

        imgs, labels = next(iter(train_loader))
        preds_raw = model.forward(imgs.to(device))
        preds = softmax(preds_raw, dim=1).cpu().detach().numpy()

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
                    plt.title(dataset_module.label_names[j] + label_pred_score + matches_label)
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

        gt_label_name = dataset.label_names[label_scalar]
        pred_label_name = dataset.label_names[pred.argmax()]

        j = 0
        for name, cs in method_cams_dict.items():
            c = cs['overlayed'][label_scalar]
            plt.subplot(3, 3, j + 2)
            plt.imshow(c)
            label_pred_score = f': {preds[i, j]:.2f}'
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

    ''' LIME '''
    if False:
        import lime
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        from skimage.segmentation import mark_boundaries

        # how to install skimage

        #
        imgs, labels = next(iter(train_loader))
        preds_raw = model.forward(imgs.to(device))
        preds = softmax(preds_raw, dim=1).cpu().detach().numpy()
        i = 0
        img = imgs[i:i + 1]
        label = labels[i:i + 1]
        pred = preds[i]
        label_scalar = label[0].item()  # label 0D
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)  # img_np 3D

        import matplotlib.pyplot as plt
        from PIL import Image
        import torch.nn as nn
        import numpy as np
        import os
        import json

        import torch
        from torchvision import models, transforms
        from torch.autograd import Variable
        import torch.nn.functional as F


        def get_pil_transform():
            transf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224)
            ])

            return transf


        def get_preprocess_transform():
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transf = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            return transf


        def batch_predict(images):
            model.eval()
            print(f'{images.shape=}')
            batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
            print(f'{batch=}')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            batch = batch.to(device)

            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()


        pill_transf = get_pil_transform()
        preprocess_transform = get_preprocess_transform()

        explainer = lime_image.LimeImageExplainer()

        img_for_explainer = np.array(pill_transf(img[0]))
        img_t = img_for_explainer.transpose(2, 0, 1)

        """
        Start here: 
        
        - LIME did not work:    input image must be C, W, H in the pill_transf,
                                but the explain_instance expects channels_last
                                .
        Two options:                            
        1) Just run the origal .ipynb
        
        2) DIY                        
        error in rgb2xyz
        arr = _prepare_colorarray(rgb, channel_axis=-1).copy()
        ValueError: the input array must have size 3 along `channel_axis`, got (3, 224, 224)
    
        - try debugging to see what is the dimension of the image at the time of the call
        - call again with a different image channel order
        
        """
        explanation = explainer.explain_instance(img_t,
                                                 batch_predict,  # classification function
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=False)
        img_boundry1 = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry1)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry2)

        ''' Playing with lime code to debug '''
        if False:
            from skimage.color import gray2rgb
            # sklearn
            import sklearn


            class ImageExplanation(object):
                def __init__(self, image, segments):
                    """Init function.

                    Args:
                        image: 3d numpy array
                        segments: 2d numpy array, with the output from skimage.segmentation
                    """
                    self.image = image
                    self.segments = segments
                    self.intercept = {}
                    self.local_exp = {}
                    self.local_pred = {}
                    self.score = {}

                def explain_instance(self, image, classifier_fn, labels=(1,),
                                     hide_color=None,
                                     top_labels=5, num_features=100000, num_samples=1000,
                                     batch_size=10,
                                     segmentation_fn=None,
                                     distance_metric='cosine',
                                     model_regressor=None,
                                     random_seed=None,
                                     progress_bar=True):
                    """Generates explanations for a prediction.

                    First, we generate neighborhood data by randomly perturbing features
                    from the instance (see __data_inverse). We then learn locally weighted
                    linear models on this neighborhood data to explain each of the classes
                    in an interpretable way (see lime_base.py).

                    Args:
                        image: 3 dimension RGB image. If this is only two dimensional,
                            we will assume it's a grayscale image and call gray2rgb.
                        classifier_fn: classifier prediction probability function, which
                            takes a numpy array and outputs prediction probabilities.  For
                            ScikitClassifiers , this is classifier.predict_proba.
                        labels: iterable with labels to be explained.
                        hide_color: If not None, will hide superpixels with this color.
                            Otherwise, use the mean pixel color of the image.
                        top_labels: if not None, ignore labels and produce explanations for
                            the K labels with highest prediction probabilities, where K is
                            this parameter.
                        num_features: maximum number of features present in explanation
                        num_samples: size of the neighborhood to learn the linear model
                        batch_size: batch size for model predictions
                        distance_metric: the distance metric to use for weights.
                        model_regressor: sklearn regressor to use in explanation. Defaults
                        to Ridge regression in LimeBase. Must have model_regressor.coef_
                        and 'sample_weight' as a parameter to model_regressor.fit()
                        segmentation_fn: SegmentationAlgorithm, wrapped skimage
                        segmentation function
                        random_seed: integer used as random seed for the segmentation
                            algorithm. If None, a random integer, between 0 and 1000,
                            will be generated using the internal random number generator.
                        progress_bar: if True, show tqdm progress bar.

                    Returns:
                        An ImageExplanation object (see lime_image.py) with the corresponding
                        explanations.
                    """
                    if len(image.shape) == 2:
                        image = gray2rgb(image)
                    if random_seed is None:
                        random_seed = self.random_state.randint(0, high=1000)

                    if segmentation_fn is None:
                        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                                max_dist=200, ratio=0.2,
                                                                random_seed=random_seed)
                    segments = segmentation_fn(image)

                    fudged_image = image.copy()
                    if hide_color is None:
                        for x in np.unique(segments):
                            fudged_image[segments == x] = (
                                np.mean(image[segments == x][:, 0]),
                                np.mean(image[segments == x][:, 1]),
                                np.mean(image[segments == x][:, 2]))
                    else:
                        fudged_image[:] = hide_color

                    top = labels

                    data, labels = self.data_labels(image, fudged_image, segments,
                                                    classifier_fn, num_samples,
                                                    batch_size=batch_size,
                                                    progress_bar=progress_bar)

                    distances = sklearn.metrics.pairwise_distances(
                        data,
                        data[0].reshape(1, -1),
                        metric=distance_metric
                    ).ravel()


            def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                                   num_features=5, min_weight=0.):
                """Init function.

                Args:
                    label: label to explain
                    positive_only: if True, only take superpixels that positively contribute to
                        the prediction of the label.
                    negative_only: if True, only take superpixels that negatively contribute to
                        the prediction of the label. If false, and so is positive_only, then both
                        negativey and positively contributions will be taken.
                        Both can't be True at the same time
                    hide_rest: if True, make the non-explanation part of the return
                        image gray
                    num_features: number of superpixels to include in explanation
                    min_weight: minimum weight of the superpixels to include in explanation

                Returns:
                    (image, mask), where image is a 3d numpy array and mask is a 2d
                    numpy array that can be used with
                    skimage.segmentation.mark_boundaries
                """
                if label not in self.local_exp:
                    raise KeyError('Label not in explanation')
                if positive_only & negative_only:
                    raise ValueError("Positive_only and negative_only cannot be true at the same time.")
                segments = self.segments
                image = self.image
                exp = self.local_exp[label]
                mask = np.zeros(segments.shape, segments.dtype)
                if hide_rest:
                    temp = np.zeros(self.image.shape)
                else:
                    temp = self.image.copy()
                if positive_only:
                    fs = [x[0] for x in exp
                          if x[1] > 0 and x[1] > min_weight][:num_features]
                if negative_only:
                    fs = [x[0] for x in exp
                          if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
                if positive_only or negative_only:
                    for f in fs:
                        temp[segments == f] = image[segments == f].copy()
                        mask[segments == f] = 1
                    return temp, mask
                else:
                    for f, w in exp[:num_features]:
                        if np.abs(w) < min_weight:
                            continue
                        c = 0 if w < 0 else 1
                        mask[segments == f] = -1 if w < 0 else 1
                        temp[segments == f] = image[segments == f].copy()
                        temp[segments == f, c] = np.max(image)
                    return temp, mask

                ret_exp = ImageExplanation(image, segments)
                if top_labels:
                    top = np.argsort(labels[0])[-top_labels:]
                    ret_exp.top_labels = list(top)
                    ret_exp.top_labels.reverse()
                for label in top:
                    (ret_exp.intercept[label],
                     ret_exp.local_exp[label],
                     ret_exp.score[label],
                     ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                        data, labels, distances, label, num_features,
                        model_regressor=model_regressor,
                        feature_selection=self.feature_selection)

                return ret_exp

        ''' ... '''
