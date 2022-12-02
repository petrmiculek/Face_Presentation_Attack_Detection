# stdlib
import argparse
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
from torchvision.models import resnet18, ResNet18_Weights
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
import rose_youtu_dataset as dataset
from eval import confusion_matrix, compute_metrics  # , accuracy
from util import get_dict, print_dict
from model_util import EarlyStopping

"""
todo:
model:
- training: one-category, all-but-one-category
- log plots to wandb
- fix W&B 'failed to sample metrcics' error

less important:
- checkpoint also state of scheduler, optimizer, ...
- 16bit training
- setup for metacentrum

other:
- cache function calls (reading annotations)
- 

done:
- gpu training #DONE#
- eval metrics  #DONE#
- validation dataset split  #DONE#
- W&B  #DONE#
- confusion matrix  #DONE#

"""
''' Global variables '''
cat_names = list(dataset.labels.values())
num_classes = len(dataset.labels)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch size', type=int, default=config.HPARAMS['batch_size'])
parser.add_argument('--epochs', help='number of epochs', type=int, default=config.HPARAMS['epochs'])
parser.add_argument('--lr', help='learning rate', type=float, default=config.HPARAMS['lr'])
# parser.add_argument('--model', help='model name', type=str, default='resnet18')
parser.add_argument('--num_workers', help='number of workers', type=int, default=0)


# def main():
if __name__ == '__main__':
    print('Training multiclass softmax CNN on RoseYoutu')
    args = parser.parse_args()

    if "PYCHARM_HOSTED" in os.environ:
        # running in pycharm
        show_plots = True
    else:
        show_plots = False
        print('Plottting disabled')


    ''' Initialization '''
    # check available gpus
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        model = resnet18(weights=weights)
        preprocess = weights.transforms()

        # replace last layer with binary classification head
        model.fc = torch.nn.Linear(512, num_classes, bias=True)

        # freeze all previous layers
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
        bona_fide = list(dataset.labels.values()).index('genuine')  # == 0

        # get annotations (paths to samples + labels)
        paths_genuine = dataset.read_annotations('genuine')
        paths_attacks = dataset.read_annotations('attack')
        paths_all = pd.concat([paths_genuine, paths_attacks])

        '''
        Dataset contains 10 people
        Split into 8 for training, 1 for validation, 1 for testing
        Every person has the same number of samples, but not the same number of attack types
        '''
        # split subsets based on person ids
        person_ids = pd.unique(paths_all['id0'])
        val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
        train_ids = np.setdiff1d(person_ids, [val_id, test_id])

        # split train/val/test (based on person IDs)
        paths_train = paths_all[paths_all['id0'].isin(train_ids)]
        paths_val = paths_all[paths_all['id0'].isin([val_id])]
        paths_test = paths_all[paths_all['id0'].isin([test_id])]

        # shuffle order
        paths_train = paths_train.sample(frac=1).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1).reset_index(drop=True)

        # limit size for prototyping
        limit = 3200
        paths_train = paths_train[:limit]
        paths_val = paths_val[:limit]
        paths_test = paths_test[:limit]

        # dataset loaders
        loader_kwargs = {'num_workers': args.num_workers, 'batch_size': args.batch_size}
        train_loader = dataset.RoseYoutuLoader(paths_train, **loader_kwargs, shuffle=True)
        val_loader = dataset.RoseYoutuLoader(paths_val, **loader_kwargs)
        test_loader = dataset.RoseYoutuLoader(paths_test, **loader_kwargs)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)


    ''' Logging '''
    wb.config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": str(optimizer),
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

    # sample prediction
    if False:
        img, x, label = next(iter(train_loader))
        img_batch = preprocess(img)

        model.eval()

        with torch.no_grad():
            out = model(img_batch)

        pred = out.softmax(dim=1)
        prediction_hard = torch.argmax(pred, dim=1).numpy()
        category_name = cat_names[prediction_hard]
        score = pred[range(len(prediction_hard)), prediction_hard]

        # plot predictions
        for i in range(img.shape[0]):
            # convert channels order CWH -> HWC
            plt.imshow(img[i].permute(1, 2, 0))
            plt.title(f'Prediction: {category_name[i]}, Score: {score[i]:.2f}')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()

    ''' Training '''
    # run training
    best_loss_val = np.inf
    best_res = None

    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    len_test_loader = len(test_loader)

    epochs_trained = 0
    for epoch in range(args.epochs):
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

                # print(out.softmax(dim=1).detach().numpy(), y.detach().numpy(), f'loss: {loss.item():.2f}')
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
        ep_train_loss = ep_train_loss / len_train_loader
        ep_loss_val = ep_loss_val / len_val_loader

        metrics_val = compute_metrics(labels_val, preds_val)

        # log results
        res = {'Loss Training': ep_train_loss,
               'Loss Validation': ep_loss_val,
               'Accuracy Training': metrics_train['accuracy'],
               'Accuracy Validation': metrics_val['accuracy'],
               }
        wb.log(res, step=epoch)

        # print results
        print_dict(res)

        # save best results
        if ep_loss_val < best_loss_val:
            best_loss_val = ep_loss_val
            # save a deepcopy of res to best_res
            best_res = deepcopy(res)

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

    print(f'Test loss: {loss_test:.2f}')
    print(f'Test accuracy: {metrics_test["accuracy"]:.2f}')

    # log results
    res = {
        'Loss Test': loss_test,
        'Accuracy Test': metrics_test["accuracy"],
    }
    wb.log(res, step=epoch)
    wb.run.summary['loss_test'] = loss_test
    wb.run.summary['accu_test'] = metrics_test["accuracy"]

    # plot results
    preds_test = np.concatenate(preds_test)
    labels_test = np.concatenate(labels_test)
    cm = confusion_matrix(labels_test, preds_test, labels=cat_names, normalize=False, title_suffix='Test',
                     output_location=outputs_dir, show=show_plots)
    # log confusion matrix to wandb
    wb.log({"confusion_matrix": cm})






