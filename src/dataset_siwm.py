# stdlib
import logging
import os
from os.path import join
import re

# external
# import torch
# import torchvision
# from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True

# from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage
from torch.utils.data import DataLoader, Dataset

# local
# import config
from dataset_base import BaseDataset, StandardLoader

''' Hardcoded dataset paths '''
data_root_dir = '/mnt/sdb1/dp/siw_m_dataset'  # todo change
samples_dir = join(data_root_dir, 'images')
annotations_path = join(data_root_dir, 'table', 'table.csv')

''' Hardcoded labels to remember the structure '''
# order chosen accordingly to the annotations
spoof_attack_names = [
    'Makeup',
    'Live',
    'Paper',
    'Mask_partial',
    'Replay',
    'Mask'
]

spoof_attack_to_numbers = {label: i for i, label in enumerate(spoof_attack_names)}
spoof_numbers_to_attack = {v: k for k, v in spoof_attack_to_numbers.items()}

spoof_info_names = [
    'Live',
    'Makeup_Co',
    'Makeup_Im',
    'Makeup_Ob',
    'Mask_Half',
    'Mask_Mann',
    'Mask_Paper',
    'Mask_Silicone',
    'Mask_Trans',
    'Paper',
    'Partial_Eye',
    'Partial_Funnyeye',
    'Partial_Mouth',
    'Partial_Paperglass',
    'Replay'
]
spoof_info_to_numbers = {label: i for i, label in enumerate(spoof_info_names)}
spoof_numbers_to_info = {v: k for k, v in spoof_info_to_numbers.items()}

spoof_info_to_attack = {
    'Makeup_Co': 0,
    'Makeup_Im': 0,
    'Makeup_Ob': 0,
    'Partial_Paperglass': 0,
    'Live': 1,
    'Paper': 2,
    'Partial_Funnyeye': 3,
    'Partial_Eye': 3,
    'Partial_Mouth': 3,
    'Replay': 4,
    'Mask_Half': 5,
    'Mask_Mann': 5,
    'Mask_Paper': 5,
    'Mask_Silicone': 5,
    'Mask_Trans': 5
}

"""
Mapping between the granularity of '_attack' and '_info'
spoof_attack : spoof_info
0 ['Makeup_Co', 'Makeup_Im', 'Makeup_Ob', 'Partial_Paperglass']
1 ['Live']
2 ['Paper']
3 ['Partial_Funnyeye', 'Partial_Eye', 'Partial_Mouth']
4 ['Replay']
5 ['Mask_Half', 'Mask_Mann', 'Mask_Paper', 'Mask_Silicone', 'Mask_Trans']
"""


def read_annotations(f):
    annotations = pd.read_csv(annotations_path)

    """
    There are 6 spoof_attack and 15 spoof_info classes.
    """

    use_spoof_attack = True  # else use spoof_info
    label_key = 'spoof_attack' if use_spoof_attack else 'spoof_info'

    # rename columns to fit the base class (copy columns)
    # annotations.rename(columns={'image_path': 'path', label_key: 'label'}, inplace=True)
    annotations['path'] = annotations['image_path']
    annotations['label'] = annotations[label_key]

    ''' Adjust image paths '''
    paths = annotations['path']
    path_prefix = '../'
    idx_cut = paths.iloc[0].find(path_prefix) + len(path_prefix)
    annotations['path'] = [join(data_root_dir, p[idx_cut:]) for p in paths]

    ''' Remove annotations with missing images '''
    missing = []
    for i, p in enumerate(annotations['path']):
        if not os.path.isfile(p):
            missing.append(i)
            if len(missing) <= 5:
                print(p + ' does not exist')
                if len(missing) == 5:
                    print('skipping further missing files ...')

    if len(missing) > 0:
        print(f'{len(missing)} missing files from annotations will not be used')
        annotations.drop(axis=0, index=missing, inplace=True)

    ''' Convert labels to numbers '''
    if not use_spoof_attack:  # spoof_attack is already a number
        annotations['label'] = annotations['label'].apply(lambda x: spoof_info_to_numbers[x])

    return annotations


class SIWMDataset(BaseDataset):
    pass


def SIWMLoader(annotations, **kwargs):
    return StandardLoader(SIWMDataset, annotations, **kwargs)


def plot_sample_images(annotations):
    """ Plot first image of each category (spoof_info) """
    unique_labels = annotations['spoof_info'].unique().tolist()  # 15 categories

    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(unique_labels):
        plt.subplot(5, 3, i + 1)
        image = Image.open(annotations[annotations['spoof_info'] == label]['image_path'].iloc[0])
        plt.imshow(image)
        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    annotations = read_annotations(None)
    loader = SIWMLoader(annotations, batch_size=4)

    ''' Check reading from DataLoader '''
    for i, (x, y) in enumerate(loader):
        print(x.shape, y.shape)
        if i > 1:
            break

    ''' Exploring the dataset '''
    if False:
        ''' Count unique labels '''
        value_counts = annotations['spoof_attack'].value_counts()
        print(value_counts)
        value_counts = annotations['spoof_info'].value_counts()
        print(value_counts)

        ''' Plot sample image '''
        sample = annotations.iloc[4]
        image = Image.open(sample['image_path'])
        plt.imshow(image)
        plt.show()

        ''' Unique column values '''
        for k in annotations.columns:
            print(k, annotations[k].unique(), "\n\n")

        ''' Group labels by spoof_attack '''
        labels = annotations[['spoof_attack', 'spoof_info']]
        labels_attack_to_info = labels.groupby('spoof_attack').agg(lambda x: x.unique().tolist())
        info_to_attack = {}
        for attack, info_many in labels_attack_to_info.itertuples():
            print(attack, info_many)
            info_to_attack.update({info: attack for info in info_many})
