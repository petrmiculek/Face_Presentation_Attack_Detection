# stdlib
import argparse
import logging
import os
import datetime
from os.path import join


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
import wandb as wb

from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True

# local
import config
import rose_youtu_dataset as dataset
from eval import accuracy, confusion_matrix
from util import get_dict

output_categories = np.array(['genuine', 'attack'])

"""
todo:
model:
- gpu
- eval metrics
- validation dataset split
- W&B
- 

other:
- cache function calls (reading annotations)
- 
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=False)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=False)

args_global = None

num_classes = len(dataset.labels)

if __name__ == '__main__':
    args_global = parser.parse_args()


    print('Training multiclass softmax CNN on RoseYoutu')

    ''' Initialization '''
    # check available gpus
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # set logging level
    logging.basicConfig(level=logging.INFO)

    training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    outputs_dir = join('runs', training_run_id)
    checkpoint_path = join(outputs_dir, f'checkpoint_{training_run_id}.pt')

    epochs_trained = 0

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
                param.requires_grad = False
            else:
                param.requires_grad = True
            # print(name, param.requires_grad)

        model.to(device)

    ''' Dataset '''
    if True:
        """
        # approach 1: preset train/test split from annotations - stupid because it's genuine x attacks
        train_loader = dataset.RoseYoutuLoader(which='train', batch_size=4)
        test_loader = dataset.RoseYoutuLoader(which='test', batch_size=16)
        """

        # approach 2: split dataset manually
        paths_genuine = dataset.read_annotations(dataset.annotations_train_path, dataset.samples_train_dir)
        paths_attacks = dataset.read_annotations(dataset.annotations_test_path, dataset.samples_test_dir)

        len_g = len(paths_genuine)
        len_a = len(paths_attacks)
        train_split, val_split, test_split = 0.8, 0.1, 0.1
        idx_g_tr, idx_g_val = int(len_g * train_split), int(len_g * (train_split + val_split))
        idx_a_tr, idx_a_val = int(len_a * train_split), int(len_a * (train_split + val_split))

        # concatenate genuine and attacks
        print('Running on limited datset size')
        paths_training = pd.concat([paths_genuine[:idx_g_tr], paths_attacks[:idx_a_tr]])[:100]
        paths_validation = pd.concat([paths_genuine[idx_g_tr:idx_g_val], paths_attacks[idx_a_tr:idx_a_val]])[:20]
        paths_test = pd.concat([paths_genuine[idx_g_val:], paths_attacks[idx_a_val:]])[:20]

        train_loader = dataset.RoseYoutuLoader(paths_training, batch_size=4)
        val_loader = dataset.RoseYoutuLoader(paths_validation, batch_size=4)
        test_loader = dataset.RoseYoutuLoader(paths_test, batch_size=16)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

    ''' Model Setup '''
    lr = 1e-3
    batch_size = 4

    # loss optimizer etc
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    ''' Logging '''
    wb.config = {
        "run_id": training_run_id,
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": str(optimizer),
    }

    config_dump = get_dict(config)
    wb.config.update(config_dump)
    # global args_global
    args_dict = get_dict(args_global)
    wb.config.update(args_dict)
    # wb.init(project="facepad", config=wb.config)

    # sample prediction
    if False:
        img, x, label = next(iter(train_loader))
        img_batch = preprocess(img)

        model.eval()

        with torch.no_grad():
            out = model(img_batch)

        pred = out.softmax(dim=1)
        prediction_hard = torch.argmax(pred, dim=1).numpy()
        category_name = output_categories[prediction_hard]
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

    for epoch in range(3):
        print(f'Epoch {epoch}')
        model.train()
        ep_train_loss = 0
        ep_correct_train = 0
        preds_train = []
        labels_train = []

        with tqdm(train_loader, total=len(train_loader), leave=True) as progress_bar:
            for img, label in progress_bar:
                # prediction
                img = img.to(device, non_blocking=True, dtype=torch.float)
                label = label.to(device, non_blocking=True)  # , dtype=torch.float
                # torch.LongTensor

                img_batch = preprocess(img)
                out = model(img_batch)
                print(out.shape, label.shape)
                loss = criterion(out, label)
                ep_train_loss += loss.item()

                # learning step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute accuracy
                prediction_hard = torch.argmax(out, dim=1)
                match = prediction_hard == label
                ep_correct_train += match.sum().item()
                progress_bar.set_postfix(loss=f'{ep_train_loss:.4f}')

                # save predictions
                preds_train.append(prediction_hard.cpu().numpy())
                labels_train.append(label.cpu().numpy())

            # print(out.softmax(dim=1).detach().numpy(), y.detach().numpy(), f'loss: {loss.item():.2f}')

        # validation loop
        model.eval()
        with torch.no_grad():
            ep_loss_val = 0
            ep_correct_val = 0
            for img, label in tqdm(val_loader):
                img, label = img.to(device), label.to(device)
                img_batch = preprocess(img)
                out = model(img_batch)
                loss = criterion(out, label)
                ep_loss_val += loss.item()

                # compute accuracy
                prediction_hard = torch.argmax(out, dim=1)
                match = prediction_hard == label
                ep_correct_val += match.sum().item()

        ep_train_loss /= len(train_loader)  # loss is averaged over batch already
        ep_accu_train = ep_correct_train / len_train_ds
        ep_loss_val /= len(val_loader)
        ep_accu_val = ep_correct_val / len_val_ds

        # log results
        res = {'Loss Training': ep_train_loss,
               'Loss Validation': ep_loss_val,
               'Accuracy Training': ep_accu_train,
               'Accuracy Validation': ep_accu_val,
               }

        print('')  # newline
        for k, v in res.items():
            print(f'{k}: {v:.4f}')

        # wb.log(res, step=epoch)

    # model checkpointing



    # test eval
    model.eval()
    preds_test = []
    labels_test = []
    with torch.no_grad():
        total_loss_test = 0
        total_correct_test = 0
        for img, label in tqdm(test_loader):
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
            preds_test.append(prediction_hard.cpu().numpy())
            labels_test.append(label.cpu().numpy())

        loss_test = total_loss_test / len(test_loader)
        accu_test = total_correct_test / len_test_ds
        print(f'Test loss: {loss_test:.2f}')
        print(f'Test accuracy: {accu_test:.2f}')

    # log results
    res = {
        'Loss Test': loss_test,
        'Accuracy Test': accu_test,
    }
    wb.log(res, step=epoch)

    # save model
    torch.save(model.state_dict(), checkpoint_path)

    # plot results
    preds_test = np.concatenate(preds_test)
    labels_test = np.concatenate(labels_test)
    plot_confusion_matrix(labels_test, preds_test, output_categories, normalize=True)

