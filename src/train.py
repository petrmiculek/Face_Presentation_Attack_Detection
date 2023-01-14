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
import torch
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import ResNet18_Weights
import numpy as np
from tqdm import tqdm
import pandas as pd

os.environ["WANDB_SILENT"] = "true"

import wandb as wb

from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
from metrics import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, keys_append, save_dict_json
from model_util import EarlyStopping
import resnet18
from dataset_base import pick_dataset_version, load_dataset

"""
todo:
prio:
- 

normal:
- training: one_attack, unseen_attack
    - log the class chosen for training/val/test  #DONE#
    - include the class names in the confusion matrix title

- extract one-to-last layer embeddings (for t-SNE etc.)
    - resnet18.py - local model implementation

- log the plots to wandb

less important:
- setup for metacentrum
- fix W&B 'failed to sample metrics' error
- 16bit training
- checkpoint also state of scheduler, optimizer, ...

other:
- cache function calls (reading annotations?)
- 

done:
- gpu training #DONE#
- eval metrics  #DONE#
- validation dataset split  #DONE#
- W&B  #DONE#
- confusion matrix  #DONE#

notes:

one_attack training: 
    - train genuine + one type of attack  #DONE#
    - binary predictions  #DONE#
    - shuffling the dataset  #DONE#
    - mixing genuine and attack data  #DONE#
    - script: run training on every category separately TODO
    + possibly include out-of-distribution data for testing?

unseen_attack training:
    - train genuine + 6/7 categories  #DONE#
    - test on the last category  #DONE#
    - script: run for every category as unseen

todo what is a good val x test split for one-attack
"""

''' Global variables '''
# -

''' Parsing Arguments '''
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=config.HPARAMS['batch_size'])
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=config.HPARAMS['epochs'])
parser.add_argument('-l', '--lr', help='learning rate', type=float, default=config.HPARAMS['lr'])
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str,
                    default='one_attack')
parser.add_argument('-d', '--dataset', help='dataset to train on ', type=str, default='rose_youtu')

# print('main is not being run')  # uncomment this when using def main...
# def main():
if __name__ == '__main__':

    ''' Parse arguments '''
    args = parser.parse_args()

    ''' Dataset + Training mode '''

    # load dataset module
    if args.dataset == 'rose_youtu':
        import dataset_rose_youtu as dataset_module
    elif args.dataset == 'siwm':
        import dataset_siwm as dataset_module
    else:
        raise ValueError(f'Unknown dataset name {args.dataset}')

    # set training mode
    training_mode = args.mode  # 'one_attack' or 'unseen_attack' or 'all_attacks'
    # separate_person_ids = False  # unused: train on one person, test on another
    if training_mode == 'all_attacks':
        cat_names = dataset_module.label_names
        num_classes = len(dataset_module.labels)
    elif training_mode == 'one_attack':
        cat_names = ['genuine', 'attack']
        num_classes = 2
    elif training_mode == 'unseen_attack':
        cat_names = ['genuine', 'attack']
        num_classes = 2
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')

    print(f'Training multiclass softmax CNN '
          f'on {args.dataset} dataset in {training_mode} mode')

    ''' Setup Run Environment '''
    if "PYCHARM_HOSTED" in os.environ:
        # running in pycharm
        show_plots = True
    else:
        show_plots = False
        print('Not running in Pycharm, not displaying plots')

    ''' Initialization '''
    if True:
        # check available gpus
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')
        torch.backends.cudnn.benchmark = True

        # Logging Setup
        logging.basicConfig(level=logging.WARNING)

    if True:
        # training config and logging
        training_run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.training_run_id = training_run_id

        wb.config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "training_mode": training_mode,
            "num_classes": num_classes,
        }
        wb.init(project="facepad", config=wb.config)
        run_name = wb.run.name
        print(f'W&B run name: {run_name}')
        outputs_dir = join('runs', run_name)
        os.makedirs(outputs_dir, exist_ok=True)
        checkpoint_path = join(outputs_dir, f'model_checkpoint.pt')

    ''' Model '''
    if True:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18.resnet18(weights=weights, weight_class=ResNet18_Weights)

        preprocess = weights.transforms()

        # replace last layer with n-ary classification head
        model.fc = torch.nn.Linear(512, num_classes, bias=True)

        # freeze all previous layers
        print('Note: Currently not freezing any layers')
        for name, param in model.named_parameters():
            if 'fc' not in name:
                # param.requires_grad = False
                pass
            else:
                param.requires_grad = True
            # print(name, param.requires_grad)

        model.to(device)

        ''' Model Setup '''
        criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=config.HPARAMS['lr_scheduler_factor'],
                                                               patience=config.HPARAMS['lr_scheduler_patience'],
                                                               min_lr=config.HPARAMS['lr_scheduler_min_lr'],
                                                               verbose=True)
        early_stopping = EarlyStopping(patience=config.HPARAMS['early_stopping_patience'],
                                       verbose=True, path=checkpoint_path)

    ''' Dataset '''
    if True:
        dataset_meta = pick_dataset_version(args.dataset, args.mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']

        loader_kwargs = {'shuffle': True, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
                         'pin_memory': True}
        train_loader, val_loader, test_loader = load_dataset(dataset_meta, dataset_module, limit=-1,
                                                             quiet=False, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

    ''' Logging '''
    wb.config.update(
        {"run_name": run_name,
         "optimizer": str(optimizer),
         "dataset_size": len_train_ds,
         "class_train": attack_train,
         "class_val": attack_val,
         "class_test": attack_test,
         "train_ids": dataset_meta['train_ids'],
         "val_id": dataset_meta['val_id'],
         "test_id": dataset_meta['test_id'],
         })

    config_dump = get_dict(config)
    wb.config.update(config_dump)
    # global args_global
    args_dict = get_dict(args)
    wb.config.update(args_dict)

    # Print setup
    print_dict(config_dump)
    print_dict(args_dict)

    ''' Training '''
    # run training
    best_loss_val = np.inf
    best_res = None

    epochs_trained = 0

    for epoch in range(epochs_trained, epochs_trained + args.epochs):
        print(f'Epoch {epoch}')
        model.train()
        ep_train_loss = 0
        preds_train = []
        labels_train = []

        try:
            with tqdm(train_loader, leave=False, mininterval=1.) as progress_bar:
                for img, label in progress_bar:
                    # prediction
                    img = img.to(device, non_blocking=True)  # , dtype=torch.float
                    label = label.to(device, non_blocking=True)  # , dtype=torch.float / torch.LongTensor

                    img_batch = preprocess(img)
                    out = model(img_batch)
                    loss = criterion(out, label)
                    ep_train_loss += loss.item()

                    # learning step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # compute accuracy
                    prediction_hard = torch.argmax(out, dim=1)
                    match = prediction_hard == label
                    progress_bar.set_postfix(loss=f'{loss:.4f}')

                    # save predictions
                    labels_train.append(label.cpu().numpy())
                    preds_train.append(prediction_hard.detach().cpu().numpy())

        except KeyboardInterrupt:
            print('Ctrl+C stopped training')

        # compute training metrics
        preds_train = np.concatenate(preds_train)
        labels_train = np.concatenate(labels_train)

        metrics_train = compute_metrics(labels_train, preds_train)

        ''' Validation loop '''
        model.eval()
        ep_loss_val = 0.0
        preds_val = []
        labels_val = []
        with torch.no_grad():
            for img, label in tqdm(val_loader, leave=False, mininterval=1.):
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
                img_batch = preprocess(img)
                out = model(img_batch)
                loss = criterion(out, label)
                ep_loss_val += loss.item()

                # compute accuracy
                prediction_hard = torch.argmax(out, dim=1)

                # save prediction
                labels_val.append(label.cpu().numpy())
                preds_val.append(prediction_hard.cpu().numpy())

        # loss is averaged over batch already, divide by batch number
        ep_train_loss /= len_train_loader
        ep_loss_val /= len_val_loader

        metrics_val = compute_metrics(labels_val, preds_val)

        # log results
        res_epoch = {'Loss Training': ep_train_loss,
                     'Loss Validation': ep_loss_val,
                     'Accuracy Training': metrics_train['Accuracy'],
                     'Accuracy Validation': metrics_val['Accuracy'],
                     }

        # print results
        print_dict(res_epoch)

        # save best results
        if ep_loss_val < best_loss_val:
            best_loss_val = ep_loss_val
            # save a deepcopy of res to best_res
            best_res = deepcopy(res_epoch)
            best_res['epoch_best'] = epoch

        wb.log(res_epoch, step=epoch)

        epochs_trained += 1
        # LR scheduler
        scheduler.step(ep_loss_val)

        # model checkpointing
        early_stopping(ep_loss_val, model, epoch)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    ''' End of Training Loop '''
    print('Training finished')
    # load best model checkpoint
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print('Loaded model checkpoint')

    # print best val results
    print('Best results:')
    print_dict(best_res)

    ''' Test set evaluation '''
    model.eval()
    preds_test = []
    labels_test = []
    with torch.no_grad():
        total_loss_test = 0
        total_correct_test = 0
        for img, label in tqdm(test_loader, leave=False, mininterval=1.):
            img, label = img.to(device), label.to(device)
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, label)
            total_loss_test += loss.item()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)
            match = prediction_hard == label
            total_correct_test += match.sum().item()

            # save predictions
            labels_test.append(label.cpu().numpy())
            preds_test.append(prediction_hard.cpu().numpy())

        loss_test = total_loss_test / len_test_loader

    metrics_test = compute_metrics(labels_test, preds_test)

    ''' Log results '''
    print(f'\nLoss Test   : {loss_test:.2f}')
    print(f'Accuracy Test: {metrics_test["Accuracy"]:.2f}')
    print_dict(metrics_test)

    wb.run.summary['loss_test'] = loss_test
    wb.run.summary['accu_test'] = metrics_test["Accuracy"]
    metrics_test = keys_append(metrics_test, ' Test')
    wb.log(metrics_test, step=epochs_trained)

    ''' Plot results '''
    preds_test = np.concatenate(preds_test)
    labels_test = np.concatenate(labels_test)

    ''' Confusion matrix '''
    if args.mode == 'one_attack':
        attack_train_name = dataset_module.label_names[attack_train]
        attack_test_name = dataset_module.label_names[attack_test]
        title_suffix = f'Test' \
                       f'\ntrain: {attack_train_name}({attack_train}), ' \
                       f'test:{attack_test_name}({attack_test})'
    elif args.mode == 'unseen_attack':
        attack_test_name = dataset_module.label_names[args.attack_test]
        title_suffix = f'Test' \
                       f'\ntrain: all, ' \
                       f'test:{attack_test_name}({attack_test})'
    else:  # args.mode == 'all_attacks':
        title_suffix = 'Test' \
                       '\ntrain: all, ' \
                       'test: all'

    path_cm = join(outputs_dir, 'confusion_matrix' + '.pdf')
    cm = confusion_matrix(labels_test, preds_test, labels=cat_names,
                          normalize=False, title_suffix=title_suffix,
                          output_location=path_cm, show=show_plots)

    # binary confusion matrix
    if args.mode == 'all_attacks':
        path_cm = join(outputs_dir, 'confusion_matrix_binary' + '.pdf')
        cm = confusion_matrix(labels_test > 0, preds_test > 0, labels=['genuine', 'attack'],
                              normalize=False, title_suffix=title_suffix,
                              output_location=path_cm, show=show_plots)
    ''' Save config locally '''
    union_dict = {**vars(args), **wb.config, **metrics_test, **best_res}
    print_dict(union_dict)
    save_dict_json(union_dict, path=os.path.join(outputs_dir, 'config.json'))
