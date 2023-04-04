# stdlib
import logging
import os
from os.path import join

# external
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True



# local
from dataset_base import BaseDataset, StandardLoader
# import config

"""
TODO:
- meaning of x and y --> person_ID0, label_bin #DONE#
- extract other info from the filename  #DONE#
- extract info from dirname above filename  #DONE#

"""

"""
Rose Youtu Dataset
"""
name = 'rose_youtu'

labels = {
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
# label_to_nums = dict(zip(labels.keys(), range(len(labels))))
# nums_to_names = dict(zip(range(len(labels)), labels.values()))
# label_names = list(labels.values())
# label_nums = list(label_to_nums.values())
# attack_nums = [1, 2, 3, 4, 5, 6, 7]
# label_attack = list(label_to_nums.values())
# label_attack.remove(label_genuine)

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


data_root_dir = join(os.pardir, 'data', 'client')  # todo dataset rose_youtu provide "pardir" from outside
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
    try:
        with open(path, 'r') as f:
            contents_list = f.readlines()
    except FileNotFoundError:
        logging.error(f"rose_youtu dataset annotations file not found: {path}")
        raise

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
                                'label_orig': label_to_nums[label_text],
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


def read_annotations(which='genuine', use_as_label=None):
    """
    Read annotations for the Rose Youtu dataset
    :param which: dataset part: 'genuine', or 'attack'
    :param use_as_label: copy a column to the label column, convenience thing
    :return: annotations dataframe
    """

    if which == 'both':
        annotations_genuine = _read_annotations(annotations_train_path, samples_train_dir)
        annotations_attack = _read_annotations(annotations_test_path, samples_test_dir)

        if use_as_label:
            annotations_genuine['label'] = annotations_genuine[use_as_label]
            annotations_attack['label'] = annotations_attack[use_as_label]

        return annotations_genuine, annotations_attack
    else:
        if which == 'genuine':
            annotations = _read_annotations(annotations_train_path, samples_train_dir)
        elif which == 'attack':
            annotations = _read_annotations(annotations_test_path, samples_test_dir)
        else:
            raise ValueError(f"Invalid annotations requested: {which}")

        if use_as_label:
            annotations['label'] = annotations[use_as_label]

        return annotations

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


class Dataset(BaseDataset):
    pass


def Loader(annotations, **kwargs):
    return StandardLoader(Dataset, annotations, **kwargs)


# def main():
if __name__ == '__main__':
    ''' Bare Dataset without Loader'''
    paths_genuine = read_annotations('genuine', 'label_unif')
    genuine_ds = Dataset(paths_genuine)

    # show first image
    img, label = genuine_ds[0]
    plt.imshow(img)
    plt.title(f'Training image no. 0, label: {label}')
    plt.show()

    ''' Get Dataset Loaders '''
    genuine_loader = Loader(paths_genuine, batch_size=4)
    paths_attacks = read_annotations('attack', 'label_unif')
    attack_loader = Loader(paths_attacks, batch_size=4)

    imgs, ys = next(iter(genuine_loader))
    print(imgs.shape, ys.shape)

    if False:
        cm = np.random.randint(0, 1000, (8, 8))
        pcm = pd.DataFrame(cm, columns=label_names)
