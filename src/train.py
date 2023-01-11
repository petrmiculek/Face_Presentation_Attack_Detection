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
import dataset_rose_youtu as dataset
from eval import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict, xor, keys_append, save_config
from model_util import EarlyStopping
import resnet18

"""
todo:
prio:
- 

normal:
- training: one_attack, unseen_attack
    - log the class chosen for training/val/test
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
bona_fide = list(dataset.labels.values()).index('genuine')  # == 0

''' Parsing Arguments '''
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=config.HPARAMS['batch_size'])
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=config.HPARAMS['epochs'])
parser.add_argument('-l', '--lr', help='learning rate', type=float, default=config.HPARAMS['lr'])
# parser.add_argument('-d','--model', help='model name', type=str, default='resnet18')
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str,
                    default='one_attack')
# following (2) arguments are for one_attack mode only
parser.add_argument('-t', '--attack_test', help='attack type to test on (1..7), random by default', type=int, default=-1)
parser.add_argument('-r', '--attack_train', help='attack type to train on (1..7), random by default', type=int, default=-1)

# print('main is not being run')  # uncomment this when using def main...
# def main():
if __name__ == '__main__':

    ''' Parse arguments '''
    args = parser.parse_args()

    if args.mode == 'one_attack' and xor(args.attack_test == -1, args.attack_train == -1):
        raise ValueError('one_attack mode requires both or none of --attack_test --attack_train arguments')

    ''' Setup Training Mode '''
    training_mode = args.mode  # 'one_attack' or 'unseen_attack' or 'all_attacks'
    separate_person_ids = False  # if True, train on one person, test on another

    print(f'Training multiclass softmax CNN on RoseYoutu')
    print(f'Training mode: {training_mode}')
    if training_mode == 'all_attacks':
        cat_names = dataset.label_names
        num_classes = len(dataset.labels)
    elif training_mode == 'one_attack':
        cat_names = ['genuine', 'attack']
        num_classes = 2
    elif training_mode == 'unseen_attack':
        cat_names = ['genuine', 'attack']
        num_classes = 2
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')

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

        training_run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.training_run_id = training_run_id
        outputs_dir = join('runs', training_run_id)
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
        # get annotations (paths to samples + labels)
        paths_genuine, paths_attacks = dataset.read_annotations('both')
        paths_all = pd.concat([paths_genuine, paths_attacks])

        person_ids = pd.unique(paths_all['id0'])

        ''' Split dataset according to training mode '''
        label_nums = list(dataset.label_nums.values())
        if training_mode == 'all_attacks':
            ''' Train on all attacks, test on all attacks '''

            paths_all['label'] = paths_all['label_num']  # 0..7
            '''
            Dataset contains 10 people
            Split into 8 for training, 1 for validation, 1 for testing
            Every person has the same number of samples, but not the same number of attack types
            '''
            # split subsets based on person ids
            val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
            train_ids = np.setdiff1d(person_ids, [val_id, test_id])

            # split train/val/test (based on person IDs)
            paths_train = paths_all[paths_all['id0'].isin(train_ids)]
            paths_val = paths_all[paths_all['id0'].isin([val_id])]
            paths_test = paths_all[paths_all['id0'].isin([test_id])]

            class_train = class_val = class_test = 'all'

        elif training_mode == 'one_attack':
            ''' Train on one attack type, test on another '''

            # note: ignoring person ID overlap between training and testing

            """
            Figure out splitting of:
                persons w.r.t. attack and genuine
                a) train on person 1, test on person 2
                b) train on all persons, test on one unseen person
                
            """
            paths_all['label'] = paths_all['label_bin']  # 0,1

            # attack-splitting
            if args.attack_test == -1:
                # random
                class_train, class_val, class_test = np.random.choice(label_nums, size=3, replace=False)
            else:
                # specific attack
                class_train = args.attack_train
                class_test = args.attack_test
                # random other class for validation
                class_val = np.random.choice(np.setdiff1d(label_nums, [class_train, class_test]), size=None)

            # person-splitting
            val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
            train_ids = np.setdiff1d(person_ids, [val_id, test_id])

            # split train/val/test (based on attack type and person IDs)
            paths_train = paths_all[paths_all['label_num'].isin([bona_fide, class_train])
                                    & paths_all['id0'].isin(train_ids)]
            paths_val = paths_all[paths_all['label_num'].isin([bona_fide, class_val])
                                  & paths_all['id0'].isin([val_id])]
            paths_test = paths_all[paths_all['label_num'].isin([bona_fide, class_test])
                                   & paths_all['id0'].isin([test_id])]

        elif training_mode == 'unseen_attack':
            ''' Train on all attacks except one, test on the unseen attack '''

            # note: test set == validation set
            paths_all['label'] = paths_all['label_bin']  # 0,1

            # attack-splitting
            class_test = np.random.choice(label_nums, size=None, replace=False)
            class_val = class_test
            class_train = np.setdiff1d(label_nums, [class_test])

            # person-splitting
            test_id = np.random.choice(person_ids, size=None, replace=False)
            val_id = test_id
            train_ids = np.setdiff1d(person_ids, [test_id])

            # split train/val/test (based on attack type and person IDs)
            paths_train = paths_all[paths_all['label_num'].isin(class_train)
                                    & paths_all['id0'].isin(train_ids)]
            paths_test = paths_all[paths_all['label_num'].isin([bona_fide, class_test])
                                   & (paths_all['id0'] == test_id)]
            paths_val = paths_test  # note: validation == test

        ''' Safety check '''
        unique_classes = pd.concat([paths_train, paths_val, paths_test])['label'].nunique()
        assert unique_classes == num_classes, \
            f'Number of unique classes in dataset does not match number of classes in model\n' \
            f'real: {unique_classes}, expected: {num_classes}'

        # shuffle order - useful when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1).reset_index(drop=True)

        # limit size for prototyping
        limit = 640  # -1 for no limit, 3200  # TODO dataset size limit, do not forget
        paths_train = paths_train[:limit]
        paths_val = paths_val[:limit]
        paths_test = paths_test[:limit]

        print('Dataset labels per split:')
        for paths in [paths_train, paths_val, paths_test]:
            print(paths['label'].value_counts())

        # dataset loaders
        loader_kwargs = {'num_workers': args.num_workers, 'batch_size': args.batch_size}
        train_loader = dataset.RoseYoutuLoader(paths_train, **loader_kwargs, shuffle=True)
        val_loader = dataset.RoseYoutuLoader(paths_val, **loader_kwargs)
        test_loader = dataset.RoseYoutuLoader(paths_test, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

        len_train_loader = len(train_loader)
        len_val_loader = len(val_loader)
        len_test_loader = len(test_loader)

    ''' Logging '''
    wb.config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": str(optimizer),
        "dataset_size": len_train_ds,
        "training_mode": training_mode,
        "num_classes": num_classes,
        "class_train": class_train,
        "class_val": class_val,
        "class_test": class_test,
    }

    config_dump = get_dict(config)
    wb.config.update(config_dump)
    # global args_global
    args_dict = get_dict(args)
    wb.config.update(args_dict)
    wb.init(project="facepad", config=wb.config)

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
                    preds_train.append(prediction_hard.cpu().numpy())  # todo check detach

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

    # test eval
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

    print(f'\nLoss Test   : {loss_test:.2f}')
    print(f'Accuracy Test: {metrics_test["Accuracy"]:.2f}')
    print_dict(metrics_test)

    # log results
    wb.run.summary['loss_test'] = loss_test
    wb.run.summary['accu_test'] = metrics_test["Accuracy"]
    metrics_test = keys_append(metrics_test, ' Test')
    wb.log(metrics_test, step=epochs_trained)

    ''' Plot results '''
    preds_test = np.concatenate(preds_test)
    labels_test = np.concatenate(labels_test)

    # plot confusion matrix
    if args.mode == 'one_attack':
        attack_train_name = dataset.label_names[class_train]
        attack_test_name = dataset.label_names[class_test]
        title_suffix = f'Test' \
                       f'\ntrain: {attack_train_name}({class_train}), test:{attack_test_name}({class_test})'
    elif args.mode == 'unseen_attack':
        attack_test_name = dataset.label_names[args.attack_test]
        title_suffix = f'Test' \
                       f'\ntrain: all, test:{attack_test_name}({class_test})'
    else:  # args.mode == 'all_attacks':
        title_suffix = 'Test' \
                       '\ntrain: all, test: all'

    cm = confusion_matrix(labels_test, preds_test, labels=cat_names, normalize=False, title_suffix=title_suffix,
                          output_location=outputs_dir, show=show_plots)

    ''' Save config locally '''
    union_dict = {**vars(args), **wb.config, **metrics_test, **best_res}
    save_config(union_dict, path=os.path.join(outputs_dir, 'config.json'))

