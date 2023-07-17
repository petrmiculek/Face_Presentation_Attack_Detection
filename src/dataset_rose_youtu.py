#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'
# TODO Describe this file
"""
Dataset Rose-Youtu

"""

# stdlib
import logging
import os
from os.path import join

# external
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

logging.getLogger('matplotlib.font_manager').disabled = True

# local
from dataset_base import BaseDataset, StandardLoader

# import config

"""
Rose Youtu Dataset
"""
name = 'rose_youtu'

labels_orig = {
    'G': 'Genuine',  # bona-fide, no attack, 0
    'Ps': 'Printed still',
    'Pq': 'Printed quivering',
    'Vl': 'Video lenovo',
    'Vm': 'Video mac',
    'Mc': 'Mask cropped',
    'Mf': 'Mask full',
    'Mu': 'Mask upper',
    # 'Ml': 'Mask lower'  # never occurs
}

labels_unified = {
    'G': 'Genuine',  # bona-fide, no attack, 0
    'P': 'Printed',
    'V': 'Video',
    'M': 'Mask',
    'O': 'Other'
}

labels_original_to_unified = {
    'G': 'G',
    'Ps': 'P',
    'Pq': 'P',
    'Vl': 'V',
    'Vm': 'V',
    'Mc': 'M',
    'Mf': 'M',
    'Mu': 'M'
}

labels_to_unified_num = {
    # 0: 0,
    'G': 0,
    # 1: 1,
    'Ps': 1,
    # 2: 1,
    'Pq': 1,
    # 3: 2,
    'Vl': 2,
    # 4: 2,
    'Vm': 2,
    # 5: 3,
    'Mc': 3,
    # 6: 3,
    'Mf': 3,
    # 7: 3
    'Mu': 3
}

unified_to_nums = dict(zip(labels_unified.keys(), range(len(labels_unified))))
nums_to_unified = dict(zip(range(len(labels_unified)), labels_unified.values()))  # to names
label_names_unified = list(labels_unified.values())
label_nums_unified = list(unified_to_nums.values())
genuine_num_unified = bona_fide_unified = 0
attack_nums_unified = [1, 2, 3, 4]
num_classes_unified = len(labels_unified)  # 5

genuine_num = bona_fide = 0
''' Make use of old labels throw an error to prevent silent slipups '''
label_to_nums_orig = dict(zip(labels_orig.keys(), range(len(labels_orig))))
nums_to_names = dict(zip(range(len(labels_orig)), labels_orig.values()))
label_names = list(labels_orig.values())
label_nums = list(label_to_nums_orig.values())
attack_nums = [1, 2, 3, 4, 5, 6, 7]
label_attack = list(label_to_nums_orig.values())
label_attack.remove(bona_fide)

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
# ################################################
# old split of dataset rose_youtu, unused
data_root_dir = join(os.pardir, 'data', 'client')
samples_dir = join(data_root_dir, 'rgb')
samples_train_dir = join(samples_dir, 'adaptation')
samples_test_dir = join(samples_dir, 'test')
annotations_adaptation_path = join(data_root_dir, 'adaptation_list.txt')
annotations_test_path = join(data_root_dir, 'test_list.txt')
# ################################################
"""
^ dataset-split used in Zhi Li, et al., 2022, 
“One-Class Knowledge Distillation for Face Presentation Attack Detection”.

In my work, the dataset is separated differently and is used for regular training/validation/testing.
"""


def _preprocess_annotations_old(path, samples_dir):
    """
    Read annotations for the Rose Youtu dataset - internal.
    internal function, use read_annotations instead

    :param path: annotations file path
    :param samples_dir: data samples root directory
    :return: annotations dataframe
    """
    try:
        with open(path, 'r') as f:
            contents_list = f.readlines()
    except FileNotFoundError:
        logging.error(f"rose_youtu dataset annotations file not found: {path}")
        raise

    contents_list = [x.strip().split(' ') for x in contents_list]
    samples = []
    count_failed = 0
    for s in tqdm(contents_list):
        try:
            # sample row: 1/real/2_G_NT_5s_g_E_2_1/0 1 0
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
                                'label_orig': label_to_nums_orig[label_text],
                                'label_unif': labels_to_unified_num[label_text],
                                **info})
            else:
                count_failed += 1
        except:
            count_failed += 1

    if count_failed > 0:
        logging.warning(f"Missing/failed samples: {count_failed} in {samples_dir}")

    samples = pd.DataFrame(samples)
    return samples


def preprocess_annotations(path_dataset_root, use_as_label=None):
    """ Preprocess annotations file for the Rose Youtu dataset.

    paths for samples are just filenames, the path prefix gets added upon loading the dataset.
    :param path_dataset_root: path to the top-level directory of the dataset
    :return: annotations dataframe
    """
    df = pd.read_pickle(join(path_dataset_root, 'data.pkl'))
    metadata = []
    for i, r in tqdm(df.iterrows(), total=len(df)):
        p = r['path']  # absolute path
        pname = os.path.basename(p)
        pcode = pname.split('_crop')[0]
        prefix_artificial = pcode.split('_')[-1]
        pcode = f'{prefix_artificial}_{pcode}'
        metadata_sample = info_from_filename(pcode)

        label_text = metadata_sample['label_text']
        label_orig = label_to_nums_orig[label_text]
        label_unif = labels_to_unified_num[label_text]
        label_bin = int(label_unif != bona_fide_unified)  # 0 if bona-fide, 1 otherwise
        metadata.append({**metadata_sample,
                         **r.to_dict(),
                         'path': pname,
                         'label_orig': label_orig, 'label_unif': label_unif,
                         'label_bin': label_bin,
                         })
        # previous version uses `id0` from a provided annotation (_list.txt).
        # id0 does not exist for the new version.
        # id1 is bogus in the new version.

    metadata = pd.DataFrame(metadata)
    if use_as_label:
        metadata['label'] = metadata[use_as_label]

    return metadata


def preprocess_annotations_old(use_as_label=None):
    """
    Read annotations for the Rose Youtu dataset.
    :param use_as_label: copy a column to the label column, convenience thing
    :return: annotations dataframe
    """

    annotations_adaptation = _preprocess_annotations_old(annotations_adaptation_path, samples_train_dir)
    annotations_test = _preprocess_annotations_old(annotations_test_path, samples_test_dir)
    annotations = pd.concat([annotations_adaptation, annotations_test])

    if use_as_label:
        annotations['label'] = annotations[use_as_label]

    return annotations


def info_from_filename(filename):
    """
    Extract info from a filename.
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


class Dataset(BaseDataset):
    pass


def Loader(annotations, **kwargs):
    return StandardLoader(Dataset, annotations, **kwargs)


# def main():
if __name__ == '__main__':
    ''' Bare Dataset without Loader'''
    from dataset_base import load_annotations

    paths = preprocess_annotations('/mnt/sdb1/dp/rt_single', use_as_label='label_unif')
    # load_annotations
    dataset = Dataset(paths)

    # show first image
    sample = dataset[0]
    img, label = sample['image'], sample['label']
    plt.imshow(img)
    plt.title(f'Image no. 0, label: {label}')
    plt.show()

    ''' Get Dataset Loaders '''
    loader = Loader(paths, batch_size=4)

    sample = next(iter(loader))
    imgs, ys = sample['image'], sample['label']
    print(imgs.shape, ys.shape)
