# stdlib
import argparse
import datetime
import json
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
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
# from torchvision.models import resnet18  # unused, architecture loaded locally to allow changes
from torchvision.models import ResNet18_Weights
from torch import autocast, nn
from torch.cuda.amp import GradScaler
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
from src import config
from src.metrics import confusion_matrix, compute_metrics  # , accuracy
from src.util import get_dict, print_dict, keys_append, save_dict_json, count_parameters
from src.model_util import EarlyStopping
from src.resnet18 import resnet18
from src.dataset_base import pick_dataset_version, load_dataset


''' Global variables '''
# -
''' Parsing Arguments '''
# local
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='dataset to train on', type=str, default='rose_youtu')
parser.add_argument('-a', '--model', help='model name', type=str, default='resnet18')  # efficientnet_v2_s
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=config.HPARAMS['batch_size'])
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=config.HPARAMS['epochs'])
parser.add_argument('-l', '--lr', help='learning rate', type=float, default=config.HPARAMS['lr'])
parser.add_argument('-w', '--num_workers', help='number of workers', type=int, default=0)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str,
                    default='all_attacks')

# set custom help message
parser.description = 'Train a model on a dataset'

if __name__ == '__main__' and len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# print('main is not being run')  # uncomment this when using def main...
# def main():
if __name__ == '__main__':
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')

    ''' Parse arguments '''
    args = parser.parse_args()
    args_dict = get_dict(args)
    print_dict(args_dict, 'Args')
    for k, v in vars(config).items():
        # update config with args
        if k in args_dict:
            setattr(config, k, args_dict[k])
        elif k in config.HPARAMS:
            setattr(config, k, config.HPARAMS[k])

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
    label_names_binary = ['genuine', 'attack']
    # separate_person_ids = False  # unused: train on one person, test on another
    if training_mode == 'all_attacks':
        label_names = dataset_module.label_names
        num_classes = len(dataset_module.labels)
    elif training_mode == 'one_attack':
        label_names = label_names_binary
        num_classes = 2
    elif training_mode == 'unseen_attack':
        label_names = label_names_binary
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
        model_name = 'efficientnet_v2_s'
        if model_name == 'resnet18':
            # load model with pretrained weights
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18.resnet18(weights=weights, weight_class=ResNet18_Weights)
            # replace last layer with n-ary classification head
            model.fc = torch.nn.Linear(512, num_classes, bias=True)
            preprocess = weights.transforms()
        elif model_name == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model = efficientnet_v2_s(weights=weights,
                                      weight_class=EfficientNet_V2_S_Weights)
            dropout = 0.2  # as per original model code
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_classes),
            )
            # load weights
            model_name = efficientnet_v2_s.__name__
            preprocess = weights.transforms()
        else:
            raise ValueError(f'Unknown model name {model_name}')

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
        scaler = GradScaler()  # mixed precision training

    ''' Dataset '''
    if True:
        dataset_meta = pick_dataset_version(args.dataset, args.mode)
        attack_train = dataset_meta['attack_train']
        attack_val = dataset_meta['attack_val']
        attack_test = dataset_meta['attack_test']
        limit = -1
        loader_kwargs = {'shuffle': True, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
                         'pin_memory': True}
        train_loader, val_loader, test_loader = load_dataset(dataset_meta, dataset_module, limit=limit,
                                                             quiet=False, **loader_kwargs)
        bona_fide = dataset_module.bona_fide

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
         "model_name": model_name,
         })

    config_dump = get_dict(config)
    wb.config.update(config_dump)
    # global args_global

    wb.config.update(args_dict)

    # Print setup
    print_dict(config_dump, 'Config')
    count_parameters(model)

    ''' Training '''
    # run training
    best_accu_val = 0
    best_res = None

    epochs_trained = 0
    stop_training = False
    for epoch in range(epochs_trained, epochs_trained + args.epochs):
        print(f'Epoch {epoch}')
        model.train()
        ep_train_loss = 0
        preds_train = []
        labels_train = []

        try:
            with tqdm(train_loader, leave=False, mininterval=1.) as progress_bar:
                for img, label in progress_bar:
                    optimizer.zero_grad()  # set_to_none=True todo try grad zero/None
                    img = img.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    # prediction
                    with autocast(device_type='cuda', dtype=torch.float16):
                        img_batch = preprocess(img)
                        out = model(img_batch)
                        loss = criterion(out, label)

                    # backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    with torch.no_grad():
                        ep_train_loss += loss.cpu().numpy()

                        # compute accuracy
                        prediction_hard = torch.argmax(out, dim=1)
                        match = prediction_hard == label
                        progress_bar.set_postfix(loss=f'{loss:.4f}')

                        # save predictions
                        labels_train.append(label.cpu().numpy())
                        preds_train.append(prediction_hard.cpu().numpy())

        except KeyboardInterrupt:
            print('Ctrl+C stopped training')
            stop_training = True

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
                ep_loss_val += loss.cpu().numpy()

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
        if metrics_val['Accuracy'] >= best_accu_val:
            best_accu_val = metrics_val['Accuracy']
            # save a deepcopy of res to best_res
            best_res = deepcopy(res_epoch)
            best_res['epoch_best'] = epoch

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
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print('Loaded model checkpoint')

    # print best val results
    print_dict(best_res, 'Best epoch:')

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
            total_loss_test += loss.detach().cpu().numpy()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)
            match = prediction_hard == label
            total_correct_test += match.sum().detach().cpu().numpy()

            # save predictions
            labels_test.append(label.cpu().numpy())
            preds_test.append(prediction_hard.cpu().numpy())

        loss_test = total_loss_test / len_test_loader

    metrics_test = compute_metrics(labels_test, preds_test)

    ''' Log results '''
    print(f'\nLoss Test  : {loss_test:06.4f}')
    print(f'Accuracy Test: {metrics_test["Accuracy"]:06.4f}')
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
                       f'\ntrain: all but test, ' \
                       f'test:{attack_test_name}({attack_test})'
    else:  # args.mode == 'all_attacks':
        title_suffix = 'Test' \
                       '\ntrain: all, ' \
                       'test: all'

    cm_path = join(outputs_dir, 'confusion_matrix' + '.pdf')
    confusion_matrix(labels_test, preds_test, labels=label_names,
                     normalize=False, title_suffix=title_suffix,
                     output_location=cm_path, show=show_plots)

    # binary confusion matrix
    if args.mode == 'all_attacks':
        cm_binary_path = join(outputs_dir, 'confusion_matrix_binary' + '.pdf')
        confusion_matrix(labels_test != bona_fide, preds_test != bona_fide, labels=label_names_binary,
                         normalize=False, title_suffix=title_suffix,
                         output_location=cm_binary_path, show=show_plots)
    ''' Save config locally '''
    union_dict = {**vars(args), **wb.config, **metrics_test, **best_res}
    print_dict(union_dict, 'All info dump')
    save_dict_json(union_dict, path=os.path.join(outputs_dir, 'config.json'))
