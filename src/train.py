# stdlib
import logging
import os

# external
import torch
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm
import pandas as pd

from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True

# local
import config
import rose_youtu_dataset as dataset
from eval import accuracy

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

num_classes = len(dataset.labels)

if __name__ == '__main__':
    print('Training multiclass softmax CNN on RoseYoutu')

    # set logging level
    logging.basicConfig(level=logging.INFO)

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
        paths_training = pd.concat([paths_genuine[:idx_g_tr], paths_attacks[:idx_a_tr]])[:100]
        paths_validation = pd.concat([paths_genuine[idx_g_tr:idx_g_val], paths_attacks[idx_a_tr:idx_a_val]])[:20]
        paths_test = pd.concat([paths_genuine[idx_g_val:], paths_attacks[idx_a_val:]])[:20]

        train_loader = dataset.RoseYoutuLoader(paths_training, batch_size=4)
        val_loader = dataset.RoseYoutuLoader(paths_validation, batch_size=4)
        test_loader = dataset.RoseYoutuLoader(paths_test, batch_size=16)

        len_train_ds = len(train_loader.dataset)
        len_val_ds = len(val_loader.dataset)
        len_test_ds = len(test_loader.dataset)

    ''' Training '''
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

    # loss optimizer etc
    criterion = torch.nn.CrossEntropyLoss()  # softmax included in the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # run training
    model.train()

    for epoch in range(3):
        print(f'Epoch {epoch}')
        ep_train_loss = 0
        ep_correct_train = 0
        for img, label in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, label)
            ep_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)
            match = prediction_hard == label
            ep_correct_train += match.sum().item()

            # print(out.softmax(dim=1).detach().numpy(), y.detach().numpy(), f'loss: {loss.item():.2f}')
        print(f'\tTraining loss: {ep_train_loss / len(train_loader):.2f}')
        print(f'\tTraining accuracy: {ep_correct_train / len_train_ds:.2f}')


        # validation loop
        model.eval()
        with torch.no_grad():
            ep_loss_val = 0
            ep_correct_val = 0
            for img, label in tqdm(val_loader):
                img_batch = preprocess(img)
                out = model(img_batch)
                loss = criterion(out, label)
                ep_loss_val += loss.item()

                # compute accuracy
                prediction_hard = torch.argmax(out, dim=1)
                match = prediction_hard == label
                ep_correct_val += match.sum().item()

            print(f'Validation loss: {ep_loss_val / len_val_ds:.2f}')
            print(f'Validation accuracy: {ep_correct_val / len_val_ds:.2f}')

        # model checkpointing


    # test eval
    model.eval()
    with torch.no_grad():
        total_loss_test = 0
        total_correct_test = 0
        for img, label in tqdm(test_loader):
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, label)
            total_loss_test += loss.item()

            # compute accuracy
            prediction_hard = torch.argmax(out, dim=1)
            match = prediction_hard == label
            total_correct_test += match.sum().item()

        print(f'Test loss: {total_loss_test / len_test_ds:.2f}')
        print(f'Test accuracy: {total_correct_test / len_test_ds:.2f}')
