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

logging.getLogger('matplotlib.font_manager').disabled = True

from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage
from torch.utils.data import DataLoader, Dataset

# local
import config

"""
TODO:
- meaning of x and y --> person_ID0, label_bin #DONE#
- extract other info from the filename  #DONE#
- extract info from dirname above filename  #DONE#

"""

"""
Rose Youtu Dataset
"""
labels = {
    'G': 'genuine',  # bona-fide, no attack, 0
    'Ps': 'printed still',
    'Pq': 'printed quivering',
    'Vl': 'video lenovo',
    'Vm': 'video mac',
    'Mc': 'mask cropped',
    'Mf': 'mask full',
    'Mu': 'Mask upper',
    # 'Ml': 'Mask lower'  # never occurs
}

label_nums = dict(zip(labels.keys(), range(len(labels))))
label_names = list(labels.values())

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
samples_dir = join(data_root_dir, 'rgb')
samples_train_dir = join(samples_dir, 'adaptation')
samples_test_dir = join(samples_dir, 'test')
adaptation_txt = 'adaptation_list.txt'
test_txt = 'test_list.txt'
annotations_train_path = join(data_root_dir, adaptation_txt)
annotations_test_path = join(data_root_dir, test_txt)


def _read_annotations(path, samples_dir):
    """
    Read annotations for the Rose Youtu dataset
    internal function, use read_annotations instead

    :param path: annotations file path
    :param samples_dir: data samples root directory
    :return: annotations dataframe
    """
    with open(path, 'r') as f:
        contents_list = f.readlines()
    contents_list = [x.strip().split(' ') for x in contents_list]
    samples = []
    count_failed = 0
    for s in contents_list:
        try:
            path = join(samples_dir, s[0] + '.jpg')

            # skip non-existing samples
            if os.path.isfile(path):
                label_dir, code = s[0].split('/')[1:3]
                info = info_from_filename(code)
                label_text = info['label_text']
                samples.append({'path': path,
                                'id0': int(s[1]),
                                'label_bin': int(s[2]),
                                'label_dir': label_dir,
                                'label_num': label_nums[label_text],
                                **info})
            else:
                count_failed += 1
        except:
            count_failed += 1

    if count_failed > 0:
        logging.warning(f"Missing/failed samples: {count_failed} in {samples_dir}")

    samples = pd.DataFrame(samples)
    return samples


def read_annotations(which='genuine', use_as_label=None):
    """
    Read annotations for the Rose Youtu dataset
    :param which: dataset part: 'genuine', or 'attack'
    :param use_as_label: copy a column to the label column, convenience thing
    :return: annotations dataframe
    """
    if which == 'genuine':
        annotations = _read_annotations(annotations_train_path, samples_train_dir)
    elif which == 'attack':
        annotations = _read_annotations(annotations_test_path, samples_test_dir)
    else:
        raise ValueError(f"Invalid annotations requested: {which}")

    if use_as_label:
        annotations['label'] = annotations[use_as_label]

    return annotations

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

        return image, sample['label']


def RoseYoutuLoader(annotations, **kwargs):
    """
    Returns a dataloader for the Rose Youtu dataset
    :param annotations: DataFrame with list of files
    :param kwargs: keyword arguments for DataLoader
    :return: dataloader

    possibly add:
    - fraction to limit dataset size
    - random seed + fixed
    - ddp
    - drop_last
    -

    """
    shuffle = kwargs.pop('shuffle', False)
    batch_size = kwargs.pop('batch_size', 1)
    num_workers = kwargs.pop('num_workers', 1)

    kwargs_dataset = {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'pin_memory': True,
        'drop_last': True,
        'shuffle': shuffle
    }

    kwargs_dataset.update(kwargs)

    transform = Compose([
        # Resize((224, 224)),
        ToTensor(),  # transforms.PILToTensor(),
        # ConvertImageDtype(torch.float),
        # Normalize(mean=[0.485, 0.456, 0.406],
    ])

    dataset = RoseYoutuDataset(annotations, transform=transform)
    loader = DataLoader(dataset, **kwargs_dataset)

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
    ID_L_S_D_x_E_p_N
    ID: person ID (1-10)
    L: label
    S: speaking
    D: device
    x: eyeglasses
    E: background environment (unused)
    p: person ID2 (matches the 1-25 numbering in the larger version of this dataset)
    N: index number
    
    """
    keys = ['id1', 'label_text', 'speaking', 'device', 'glasses', 'environment', 'id2', 'idx']
    values = filename.split('_')
    assert len(values) == len(keys), f"Expected {len(keys)} values, got {len(values)}"

    return dict(zip(keys, values))


# def main():
if __name__ == '__main__':
    ''' Bare Dataset without Loader'''
    paths_genuine = read_annotations('genuine', 'label_num')
    genuine_ds = RoseYoutuDataset(paths_genuine)

    # show first image
    img, label = genuine_ds[0]
    plt.imshow(img)
    plt.title(f'Training image no. 0, label: {label}')
    plt.show()

    ''' Get Dataset Loaders '''
    genuine_loader = RoseYoutuLoader(paths_genuine, batch_size=4)
    paths_attacks = read_annotations('attack', 'label_num')
    attack_loader = RoseYoutuLoader(paths_attacks, batch_size=4)

    imgs, ys = next(iter(genuine_loader))
    print(imgs.shape, ys.shape)
