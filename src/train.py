# stdlib
import logging
import os

# external
import torch
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True

# local
import config
import rose_youtu_dataset

output_categories = np.array(['genuine', 'attack'])

"""
todo:
- gpu
- eval metrics
- validation dataset split
- W&B
- 

"""



if __name__ == '__main__':
    print('Starting training ...')

    # set logging level
    logging.basicConfig(level=logging.INFO)

    ''' Model '''

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    preprocess = weights.transforms()

    # replace last layer with binary classification head
    model.fc = torch.nn.Linear(512, 2, bias=True)

    # freeze all previous layers
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_graf = True
        # print(name, param.requires_grad)

    ''' Dataset '''

    train_loader = rose_youtu_dataset.RoseYoutuLoader(which='train', batch_size=4)
    test_loader = rose_youtu_dataset.RoseYoutuLoader(which='test', batch_size=4)

    ''' Training '''
    # sample prediction
    if False:
        img, x, y = next(iter(train_loader))
        img_batch = preprocess(img)

        model.eval()

        with torch.no_grad():
            out = model(img_batch)

        pred = out.softmax(dim=1)
        class_id = torch.argmax(pred, dim=1).numpy()
        category_name = output_categories[class_id]
        score = pred[range(len(class_id)), class_id]

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

    model.train()
    # train
    for epoch in range(10):
        print(f'Epoch {epoch}')
        total_loss = 0
        for img, x, y in train_loader:
            img_batch = preprocess(img)
            out = model(img_batch)
            loss = criterion(out, y)
            print(out.detach().numpy(), y.detach().numpy(), loss.item())
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        # validation loop

        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.2f}')

    # test eval
