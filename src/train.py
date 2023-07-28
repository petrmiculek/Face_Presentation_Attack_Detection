#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'

"""
Train a model on a dataset.
- main training loop
- support any training mode, dataset, model
- mixed precision, gradient accumulation
- multi-task training
- model checkpointing + early stopping
- config saving
- logging + W&B
- evaluation
"""
# stdlib
import argparse
import datetime
import logging
import os
from os.path import join
from copy import deepcopy

# fix for problems with local imports - add all local directories to python path
import sys

sys_path_extension = [os.getcwd()] + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# external
import torch

from torch import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

import wandb as wb

logging.getLogger('matplotlib.font_manager').disabled = True
# disable all matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# local
import config
from metrics import plot_confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, keys_append, save_dict_json, update_config
from util_torch import EarlyStopping, load_model, get_dataset_module, count_parameters
from dataset_base import pick_dataset_version, load_dataset, get_dataset_setup, split_dataset_name, label_names_binary
from util_torch import init_device, init_seed, eval_loop

''' Global variables '''
# -
''' Parsing Arguments '''
# local
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path to dataset', type=str, default=None)
parser.add_argument('-d', '--dataset', help='dataset to train on', type=str, default='rose_youtu')
parser.add_argument('-a', '--arch', help='model architecture', type=str, default='resnet18')  # efficientnet_v2_s
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=config.HPARAMS['batch_size'])
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=config.HPARAMS['epochs'])
parser.add_argument('-f', '--freeze', help='freeze backbone', action='store_true')
parser.add_argument('-l', '--lr', help='learning rate', type=float, default=config.HPARAMS['lr'])
parser.add_argument('-t', '--limit', help='limit dataset size', type=int, default=-1)
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str,
                    default='all_attacks')
parser.add_argument('-k', '--attack', help='attack for unseen_attack and one_attack modes  (1..C)', type=int,
                    default=None)  # falsy default value: also ''
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-n', '--no_log', help='no logging = dry run', action='store_true')
# set custom help message
parser.description = 'Train a model on a dataset'

# print('main is not being run')  # uncomment this when using def main...
# def main():
if __name__ == '__main__':
    ''' Parse arguments '''
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    args.dataset, dataset_note = split_dataset_name(args.dataset)  # 'rose_youtu-single' -> 'rose_youtu', 'single'
    args_dict = get_dict(args)
    update_config(args_dict, global_vars=False, hparams=True)

    ''' Start W&B '''
    # all stdout is saved to W&B after init
    wb.init(project="facepad", config=config.HPARAMS, mode='disabled' if args.no_log else None)

    print(f'Running: {__file__}\n'
          f'In dir: {os.getcwd()}')
    print_dict(args_dict, title='Args')

    ''' Dataset + Training mode '''
    dataset_module = get_dataset_module(args.dataset)

    # set training mode
    training_mode = args.mode  # 'one_attack' or 'unseen_attack' or 'all_attacks'
    label_names, num_classes = get_dataset_setup(dataset_module, training_mode)
    ''' Initialization '''
    if True:
        ''' (Random) seed '''
        seed = args.seed if args.seed else np.random.randint(0, 2 ** 32 - 1)
        init_seed(seed)

        # training config and logging
        training_run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.training_run_id = training_run_id

        run_name = wb.run.name
        print(f'W&B run name: {run_name}')
        if args.no_log:
            print('Not logging to W&B')
            outputs_dir = None
            checkpoint_path = None
        else:
            outputs_dir = join(config.runs_dir, run_name)
            os.makedirs(outputs_dir, exist_ok=True)
            checkpoint_path = join(outputs_dir, f'model_checkpoint.pt')

    print(f'Training multiclass softmax CNN '
          f'on {args.dataset} dataset in {training_mode} mode')

    ''' Setup Run Environment '''
    if "PYCHARM_HOSTED" in os.environ:
        # running in pycharm
        show_plots = True
    else:
        show_plots = False
        print('Not running in Pycharm, not displaying plots')

    ''' Device Info '''
    device = init_device()
    # Logging Setup
    logging.basicConfig(level=logging.WARNING)

    ''' Model '''
    if True:
        model_name = args.arch
        model, preprocess = load_model(model_name, num_classes, freeze_backbone=args.freeze)
        model.to(device)

        ''' Training Setup '''
        # loss/criterion objects - separation necessary; softmax included in the loss
        criterions = {'bin': CrossEntropyLoss(), 'unif': CrossEntropyLoss(), 'orig': CrossEntropyLoss()}
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.HPARAMS['lr_scheduler_factor'],
                                      patience=config.HPARAMS['lr_scheduler_patience'],
                                      min_lr=config.HPARAMS['lr_scheduler_min_lr'], verbose=True)
        early_stopping = EarlyStopping(patience=config.HPARAMS['early_stopping_patience'],
                                       verbose=True, path=checkpoint_path)
        scaler = GradScaler()  # mixed precision training (16-bit)

    ''' Dataset '''
    if True:
        dataset_meta = pick_dataset_version(args.dataset, args.mode, attack=args.attack, note=dataset_note)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']
        limit = args.limit if (args.limit != -1) else -1  # -1 for no limit
        loader_kwargs = {'shuffle': True, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
                         'pin_memory': True, 'seed': seed,
                         'transform_train': preprocess['train'], 'transform_eval': preprocess['eval']}
        train_loader, val_loader, test_loader = load_dataset(dataset_meta, dataset_module, path_prefix=args.path,
                                                             limit=limit, quiet=False, **loader_kwargs)
        bona_fide = dataset_module.bona_fide_unified
        len_train_ds, len_val_ds, len_test_ds = len(train_loader.dataset), len(val_loader.dataset), len(
            test_loader.dataset)
        len_train_loader, len_val_loader, len_test_loader = len(train_loader), len(val_loader), len(test_loader)

    ''' Logging '''
    # remove seed, otherwise wb.config.update errors out when overwriting with None
    args_dict.pop('seed')
    # updating with program args (overwrites config)
    wb.config.update(
        {**args_dict,
         "run_name": run_name,
         "optimizer": str(optimizer),
         "dataset_size": len_train_ds,
         "class_train": attack_train,
         "class_val": attack_val,
         "class_test": attack_test,
         "train_ids": dataset_meta['train_ids'],
         "val_ids": dataset_meta['val_ids'],
         "test_ids": dataset_meta['test_ids'],
         "num_classes": num_classes,
         "seed": seed,
         })  # , allow_val_change=True)

    # Print setup
    config_dump = config.HPARAMS  # get_dict(config)
    print('Printing config, values may have been overwritten by program args')
    print_dict(config_dump, title='Config:')
    count_parameters(model, sum_only=True)

    ''' Training '''
    # run training
    best_accu_val = 0
    best_res = None

    epochs_trained = 0
    stop_training = False
    key = 'unif' if training_mode == 'all_attacks' else 'bin'
    grad_acc_steps = 4
    for epoch in range(epochs_trained, epochs_trained + args.epochs):
        print(f'Epoch {epoch}')
        model.train()
        ep_train_loss = 0
        preds_train = []
        labels_train = []
        losses_bin, losses_mc, losses_mc_orig = [], [], []
        try:
            progress_bar = tqdm(train_loader, mininterval=1., desc=f'ep{epoch} train')
            for i, sample in enumerate(progress_bar, start=1):
                img, label = sample['image'], sample['label']
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                label_bin = sample['label_bin'].to(device, non_blocking=True)
                label_multiclass = sample['label_unif'].to(device, non_blocking=True)
                label_multiclass_orig = sample['label_orig'].to(device, non_blocking=True)
                # forward pass
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model.forward_train(img)  # prediction
                    loss_bin = criterions['bin'](out['bin'], label_bin)
                    loss_mc = criterions['unif'](out['unif'], label_multiclass)
                    loss_mc_orig = criterions['orig'](out['orig'], label_multiclass_orig)
                    loss = loss_bin + loss_mc + loss_mc_orig
                # backward pass
                scaler.scale(loss).backward()
                if i % grad_acc_steps == 0:  # gradient step with accumulated gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():  # save predictions
                    losses_bin.append(loss_bin.cpu().numpy())
                    losses_mc.append(loss_mc.cpu().numpy())
                    losses_mc_orig.append(loss_mc_orig.cpu().numpy())
                    ep_train_loss += loss.cpu().numpy()
                    prediction_hard = torch.argmax(out[key], dim=1)
                    labels_train.append(label.cpu().numpy())
                    preds_train.append(prediction_hard.cpu().numpy())

                progress_bar.set_postfix(loss=f'{loss:.4f}', refresh=False)
            # end of training epoch loop

        except KeyboardInterrupt:
            print(f'Ctrl+C stopped training')
            stop_training = True

        # compute training metrics
        preds_train, labels_train = np.concatenate(preds_train), np.concatenate(labels_train)
        metrics_train = compute_metrics(labels_train, preds_train)
        ep_train_loss /= len_train_loader
        losses_bin, losses_mc, losses_mc_orig = np.mean(losses_bin), np.mean(losses_mc), np.mean(losses_mc_orig)
        print(f'Losses: bin: {losses_bin:.4f}, mc: {losses_mc:.4f}, rose: {losses_mc_orig:.4f}')

        ''' Validation loop '''
        model.eval()
        _, labels_val, preds_val, probs_val, ep_loss_val = eval_loop(model, val_loader, criterions[key], device,
                                                                     f'ep{epoch} val')
        metrics_val = compute_metrics(labels_val, preds_val)
        # log results
        res_epoch = {'Loss Training': ep_train_loss,
                     'Loss Validation': ep_loss_val,
                     'Accuracy Training': metrics_train['Accuracy'],
                     'Accuracy Validation': metrics_val['Accuracy']}

        # print results
        print_dict(res_epoch)

        # save best results
        if metrics_val['Accuracy'] >= best_accu_val:
            best_accu_val = metrics_val['Accuracy']
            # save a deepcopy of res to best_res
            best_res = deepcopy(res_epoch)

        wb.log(res_epoch, step=epoch)

        epochs_trained += 1
        # LR scheduler
        scheduler.step(ep_loss_val)

        # model checkpointing
        early_stopping(ep_loss_val, model, epoch)
        if early_stopping.early_stop or stop_training:
            print('Early stopping')
            break

    ''' End of Training Loop '''
    print('Training finished')
    # load best model checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print('Loaded model checkpoint')

    # print best val results
    print_dict(best_res, 'Best epoch:')

    ''' Test set evaluation '''
    model.eval()
    # end of test loop
    _, labels_test, preds_test, _, loss_test = eval_loop(model, test_loader, criterions[key], device, 'Test')
    metrics_test = compute_metrics(labels_test, preds_test)
    ''' Log results '''
    print(f'\nLoss Test  : {loss_test:06.4f}')
    print(f'Accuracy Test: {metrics_test["Accuracy"]:06.4f}')
    print_dict(metrics_test)

    wb.run.summary['loss_test'] = loss_test
    wb.run.summary['accu_test'] = metrics_test["Accuracy"]
    wb.run.summary['Epoch Best'] = early_stopping.best_epoch

    metrics_test = keys_append(metrics_test, ' Test')
    metrics_test['epochs_trained'] = epochs_trained
    wb.log(metrics_test, step=epochs_trained)

    ''' Plot results - Confusion matrix '''
    if args.mode == 'one_attack':
        # attack_test_name = attack_test  # dataset_module.label_names_unified[attack_test]
        title_suffix = f'Test' \
                       f'\ntrain: {attack_train}, test: {attack_test}'
    elif args.mode == 'unseen_attack':
        # attack_test_name = attack_test  # dataset_module.label_names_unified[attack_test]
        title_suffix = f'Test' \
                       f'\ntrain: all but test, test: {attack_test}'
    else:  # args.mode == 'all_attacks':
        title_suffix = 'Test' \
                       '\ntrain: all, test: all'

    cm_path = join(outputs_dir, 'confmat_test' + '.pdf') if not args.no_log else None
    plot_confusion_matrix(labels_test, preds_test, labels=label_names,
                          title_suffix=title_suffix, output_path=cm_path)
    # binary confusion matrix
    cm_binary_path = join(outputs_dir, 'confmat_binary_test' + '.pdf') if not args.no_log else None
    plot_confusion_matrix(labels_test != bona_fide, preds_test != bona_fide, labels=label_names_binary,
                          title_suffix=title_suffix, output_path=cm_binary_path)

    ''' Save config locally '''
    union_dict = {**vars(args), **wb.config, **metrics_test, **best_res}
    print_dict(union_dict, 'All info dump')
    path_config_out = os.path.join(outputs_dir, 'config.json') if not args.no_log else None
    save_dict_json(union_dict, path=path_config_out)
