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

# local
# import config
from dataset_base import BaseDataset, StandardLoader

name = 'siwm'

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

bona_fide = 1
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
                print(f'Missing image during loading dataset SIW-M: {p}')
                if len(missing) == 5:
                    print('skipping further missing files ...')

    if len(missing) > 0:
        print(f'{len(missing)} missing files from annotations will not be used')
        annotations.drop(axis=0, index=missing, inplace=True)

    ''' Convert labels to numbers '''
    if not use_spoof_attack:  # spoof_attack is already a number
        annotations['label'] = annotations['label'].apply(lambda x: spoof_info_to_numbers[x])

    return annotations


class Dataset(BaseDataset):
    pass


def Loader(annotations, **kwargs):
    return StandardLoader(Dataset, annotations, **kwargs)


def plot_sample_images(annotations):
    """ Plot first image of each category (spoof_info) """
    unique_labels = annotations['spoof_info'].unique().tolist()  # 15 categories

    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(unique_labels):
        plt.subplot(5, 3, i + 1)
        image = Image.open(annotations[annotations['spoof_info'] == label]['path'].iloc[0])
        plt.imshow(image)
        plt.title(label)
        plt.axis('off')

    plt.suptitle('SIW-M Example Images', fontsize=20)
    plt.tight_layout()
    path = join('images_plots', 'siwm_example_images.pdf')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.show()


def remove_missing_bbox(annotations):
    """
    Remove annotations with missing bounding box

    bbox is stored as a string in the format [x1, y1, dx, dy]
    missing values have a NaN float value

    :param annotations:
    :return:
    """

    # list all unique types of bbox
    bbox_types = annotations['bbox'].apply(lambda x: type(x))

    # no bbox detected
    from copy import deepcopy
    idx_float = bbox_types[bbox_types != str].index
    no_bbox_samples = deepcopy(annotations.iloc[idx_float])
    # save no_bbox_samples to csv
    no_bbox_samples.to_csv('no_bbox_samples.csv')

    # plot spoof_info distribution for no_bbox_samples
    if False:
        value_counts = no_bbox_samples['spoof_info'].value_counts()
        print(value_counts)
        value_counts.plot.bar()
        plt.tight_layout()
        plt.show()

    # copy images with no bbox to a new folder
    if False:
        import os
        import shutil
        target_dir = '/mnt/sdb1/dp/siw_m_dataset/no_bbox_samples'
        os.makedirs(target_dir)
        for i, r in no_bbox_samples.iterrows():
            shutil.copy(r['path'], target_dir)

    # plot samples without bbox
    if False:
        it = annotations.iloc[idx_float].iterrows()
        for i, t in enumerate(it):
            j, r = t
            image = Image.open(r['path'])
            plt.imshow(image)
            plt.show()
            print(f'{i}/{len(idx_float)}: {j} {r["spoof_info"]}')
            input()
            plt.close()

    # remove indexes with no bbox
    if False:
        annotations = annotations.drop(axis=0, index=idx_float)
        annotations.to_csv('mnt/sdb1/dp/siw_m_dataset/annotations_ready.csv')

    # todo: return annotations, save them from elsewhere


if __name__ == '__main__':
    annotations = read_annotations(None)

    if False:
        ''' Plot images of given category '''
        category = 'Makeup_Im'
        annotations_makeup = annotations[annotations['spoof_info'] == category]
        for i, t in annotations_makeup.iterrows():
            j, r = t

            plt.imshow(Image.open(r['path']))
            plt.tight_layout()
            plt.show()

            if j > 2:
                break
            input()  # wait for key press

    if False:
        loader_siwm = SIWMLoader(annotations, batch_size=4, shuffle=True)

        # compare SIW-M to RoseYoutu
        from dataset_rose_youtu import Loader, read_annotations as read_annotations_rose_youtu

        annotations_rose = read_annotations_rose_youtu('attack')
        annotations_rose['label'] = annotations_rose['label_bin']
        loader_rose = Loader(annotations_rose, batch_size=4, shuffle=True)

        ''' Check reading from DataLoader with both datasets '''
        for i, (x, y) in enumerate(loader_rose):  # loader_siwm
            print(x.shape, y.shape, y)
            if i > 1:
                break

    ''' Exploring the dataset '''
    if False:
        ''' Count unique labels '''
        if False:
            value_counts = annotations['spoof_attack'].value_counts()
            print(value_counts)
            value_counts = annotations['spoof_info'].value_counts()
            print(value_counts)

        ''' Plot sample image '''
        if False:
            sample = annotations.iloc[4]
            image = Image.open(sample['image_path'])
            plt.imshow(image)
            plt.show()

        ''' Unique column values '''
        if False:
            for k in annotations.columns:
                print(k, annotations[k].unique(), "\n\n")

        ''' Group labels by spoof_attack '''
        if False:
            labels = annotations[['spoof_attack', 'spoof_info']]
            labels_attack_to_info = labels.groupby('spoof_attack').agg(lambda x: x.unique().tolist())
            info_to_attack = {}
            for attack, info_many in labels_attack_to_info.itertuples():
                print(attack, info_many)
                info_to_attack.update({info: attack for info in info_many})
            # the result is copied above, this is just to check back on the process

        ''' Read size of images '''
        if False:
            sizes = []
            for i, r in annotations.iterrows():
                image = Image.open(r['path'])
                sizes.append(image.size)
                if i % 1000 == 0:
                    print(f'{i}/{len(annotations)}')
            sizes = np.array(sizes)
            print(sizes.shape)
            print(np.unique(sizes, axis=0))

            # resolution      number of photos
            #  1920 x 1080     14060
            #  1280 x 720      590

            mask_fullhd = sizes == (1920, 1080)  # -> [[True, True] or [False, False]]
            mask_fullhd = np.all(mask_fullhd, axis=1)

            # sizes.shape  # (14650, 2)

        # parse bbox values
        annotations['bbox'] = annotations['bbox'].apply(lambda x: [float(y) for y in x[1:-1].split(',')])

        ''' Plot 2D histogram of bbox width and height '''
        if False:
            import seaborn as sns
            dx = annotations['bbox'].apply(lambda x: x[2])
            dy = annotations['bbox'].apply(lambda x: x[3])

            sns.displot(x=dx, y=dy, color='#2BA280', cbar=True)  # signature teal
            plt.title('bbox size distribution')
            plt.xlabel('width')
            plt.ylabel('height')
            plt.tight_layout()
            plt.show()

        ''' Save sample images with bounding box '''
        if False:
            bbox_dir = '/mnt/sdb1/dp/siw_m_dataset/other/bboxes'
            os.makedirs(bbox_dir)

            for i in np.random.randint(0, len(annotations), size=100):
                r = annotations.iloc[i]
                bbox = r['bbox']
                x1, y1, dx, dy = bbox
                x2, y2 = x1 + dx, y1 + dy
                name = f'annot{i}_idx{r.name}_{r["spoof_attack"]}_{r["spoof_info"]}.png'
                image = Image.open(r['path'])
                plt.imshow(image)
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r')
                plt.savefig(os.path.join(bbox_dir, name))
                plt.close()

        ''' Plot sample images '''
        # done elsewhere
        if False:
            ''' Crop out faces inside bounding box '''
            target_dir = '/mnt/sdb1/dp/siw_m_dataset/cropped'
            os.makedirs(target_dir)

            # find last generated image
            for i, r in annotations.iterrows():
                if i % 1000 == 0:
                    print(f'{i}/{len(annotations)}')
                filename = os.path.basename(r['path'])
                target_path = os.path.join(target_dir, filename)
                if not os.path.exists(target_path):
                    print(f'missing: {i} {target_path}')
                    break

            idx_failed = []
            for i, t in enumerate(annotations.iterrows()):
                if i < 8528:
                    continue
                try:
                    j, r = t
                    bbox = r['bbox']
                    path = r['path']
                    filename = os.path.basename(path)
                    # load image
                    image = Image.open(path)
                    # crop just face + some margin
                    x1, y1, dx, dy = bbox
                    x2, y2 = x1 + dx, y1 + dy
                    margin = 0.1
                    x1, y1, x2, y2 = x1 - margin * dx, y1 - margin * dy, x2 + margin * dx, y2 + margin * dy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    image = image.crop((x1, y1, x2, y2))

                    # resize image to 256x256
                    image = image.resize((256, 256))

                    # save cropped face
                    image.save(os.path.join(target_dir, filename))
                    if (i % 1000) == 0:
                        print(f'{i}/{len(annotations)}')

                except Exception as e:
                    print(e)
                    idx_failed.append(i)
