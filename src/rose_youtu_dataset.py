# stdlib
import logging
import os
from os.path import join
import re

# external
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage
from torch.utils.data import DataLoader, Dataset

# local
import config

"""
TODO:
- meaning of x and y
- extract other info from the filename
- extract info from dirname above filename

"""

"""
Rose Youtu Dataset
"""
labels = {
    'G': 'genuine',
    'Ps': 'printed still',
    'Pq': 'printed quivering',
    'Vl': 'video lenovo',
    'Vm': 'video mac',
    'Mc': 'mask cropped',
    'Mf': 'mask full',
    'Mu': 'Mask upper',
    'Ml': 'Mask lower'
}
speaking = {
    'T': 'true',  # talking
    'NT': 'false'  # not talking
}

device = {
    'HS': 'hasee',
    'HW': 'huawei',
    'IP': 'ipad',
    '5s': 'iphone 5s',
    'ZTE': 'zte'
}

glasses = {
    'g': 'glasses',
    'wg': 'without glasses'
}
# ############################################### #


data_root_dir = join(os.pardir, 'data', 'client')
adaption_txt = 'adaptation_list.txt'
test_txt = 'test_list.txt'
samples_dir = join(data_root_dir, 'rgb')
samples_train_dir = join(samples_dir, 'adaptation')
samples_test_dir = join(samples_dir, 'test')

# set loglevel debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')


def read_annotations(path, samples_dir):
    with open(path, 'r') as f:
        contents_list = f.readlines()
    contents_list = [x.strip().split(' ') for x in contents_list]
    samples = []
    count_missing = 0
    for s in contents_list:
        try:
            path = join(samples_dir, s[0] + '.jpg')
            # skip non-existing samples
            if os.path.isfile(path):
                samples.append({'path': path, 'x': int(s[1]), 'y': int(s[2])})
            else:
                count_missing += 1
        except:
            pass

    if count_missing > 0:
        logging.warning(f"Missing samples: {count_missing} in {samples_dir}")

    samples = pd.DataFrame(samples)
    return samples


class RoseYoutuDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.transform = transform
        self.samples = annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        image = Image.open(sample['path'])
        if self.transform:
            image = self.transform(image)

        return image, sample['x'], sample['y']


def RoseYoutuLoader(which='train', batch_size=None, num_workers=None):
    """
    Returns a dataloader for the Rose Youtu dataset
    :param which:
    :param batch_size:
    :param num_workers:
    :return:

    possibly add:
    - fraction to limit dataset size
    - random seed + fixed
    - ddp
    - drop_last
    -

    """

    kwargs = {
        'num_workers': (num_workers if num_workers else 1),
        'batch_size': (batch_size if batch_size else 1),
        'pin_memory': True,
        'drop_last': True
    }

    if which == 'train':
        annotations = read_annotations(join(data_root_dir, adaption_txt), samples_train_dir)
        kwargs.update({'shuffle': True})
    elif which == 'val':
        pass
    elif which == 'test':
        annotations = read_annotations(join(data_root_dir, test_txt), samples_test_dir)
        kwargs.update({'shuffle': False})
        pass
    else:
        raise ValueError(f"Unknown which: {which}")


    transform = Compose([
        # Resize((224, 224)),
        ToTensor(),  # transforms.PILToTensor(),
        # ConvertImageDtype(torch.float),
        # Normalize(mean=[0.485, 0.456, 0.406],
    ])

    dataset = RoseYoutuDataset(annotations, transform=transform)
    loader = DataLoader(dataset, **kwargs)

    return loader


def info_from_filename(filename):
    """
    Extracts info from the filename
    :param filename:
    :return: parsed info

    could be cached
    example filename: 2_G_NT_5s_g_E_2_1
    """

    """
    Format:
    L_S_D_x_E_p_N
    L: label
    S: speaking
    D: device
    x: eyeglasses
    E: background environment (unused)
    p: person ID
    N: index number
    
    """
    keys = ['label', 'speaking', 'device', 'glasses', 'environment', 'id', 'idx']
    values = filename.split('_')
    # assert len(values) == len(keys)
    return dict(zip(keys, values))


# def main():
if __name__ == '__main__':
    ''' Bare Dataset without Loader'''
    paths_training = read_annotations(join(data_root_dir, adaption_txt), samples_train_dir)
    train_ds = RoseYoutuDataset(paths_training)

    # show first image
    img, x, y = train_ds[0]
    plt.imshow(img)
    plt.title(f'Training image no. 0, x: {x}, y: {y}')
    plt.show()

    ''' Get Dataset Loaders '''
    train_loader = RoseYoutuLoader('train', batch_size=4)

    imgs, xs, ys = next(iter(train_loader))

    test_loader = RoseYoutuLoader('test')

    """
    paths_test = read_annotations(join(data_root_dir, test_txt), samples_test_dir)
    # unused
    test_ds = RoseYoutuDataset(paths_test)

    # show first image
    img, x, y = test_ds[0]
    plt.imshow(img)
    plt.show()
    """

# if __name__ == '__main__':
#     main()
