# stdlib
import argparse
import logging
import os
from os.path import join, isfile
import datetime

# external
import numpy as np
import pandas as pd

# add 'src' to the import path
import sys

sys.path.append('src')
sys.path.extend([d for d in os.listdir() if os.path.isdir(d)])

# local
import config
import dataset_rose_youtu
import dataset_siwm
from util import save_dict_json, xor
from dataset_base import BaseDataset, StandardLoader

''' Logging format '''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

''' Parsing Arguments '''
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='dataset in {rose_youtu, siwm}', type=str, required=True)
parser.add_argument('-p', '--path', help='path to dataset top-level directory', type=str, required=True)
# following (3) argument are for one_attack and unseen_attack modes
parser.add_argument('-k', '--attack_test', help='attack type to test on (1..4), random by default', type=int,
                    default=-1)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str, required=True)
parser.add_argument('-s', '--seed', help='random seed', type=int, default=None)
parser.add_argument('-n', '--no_log', help='no logging = dry run', action='store_true')

''' Create a dataset split (train, val, test) for a training mode (all_attacks, one_attack, unseen_attack) '''
'''
RoseYoutu contains 10 people. In every training mode:
Split into 8 for training, 1 for validation, 1 for testing
Every person has the same number of samples, but not the same number of attack types

Tested on:
                |  rose_youtu  |    siwm    |   rose_youtu_full
all_attacks     | training OK  |    .       |   .
unseen_attack   | training OK  |    .       |   .
one_attack      |       .      |    .       |   .

# generation failed for one_attack on metacentrum
'''

# def main():
if __name__ == '__main__':
    """
    Create dataset training/val/test splits.
    """
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    seed = args.seed if args.seed else 42
    np.random.seed(seed)
    if args.no_log:
        print('Not saving anything, dry run')

    note = ''  # arbitrary extra info: dataset length limit, ...

    training_mode = args.mode

    if args.dataset == 'rose_youtu':
        dataset = dataset_rose_youtu
    elif args.dataset == 'siwm':
        dataset = dataset_siwm
    else:
        raise ValueError('dataset must be in {rose_youtu, siwm}')

    bona_fide = 0  # todo does not apply for siwm, also read this from dataset [func] [warn]

    if training_mode == 'all_attacks':
        label_names = dataset.label_names_unified
        num_classes = len(dataset.labels_unified)
    elif training_mode == 'one_attack':
        label_names = ['genuine', 'attack']
        num_classes = 2
    elif training_mode == 'unseen_attack':
        label_names = ['genuine', 'attack']
        num_classes = 2
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')

    ''' Read annotations (paths to samples + labels) '''
    paths = dataset.preprocess_annotations(args.path)

    id_key = 'id2'
    paths[id_key] = paths[id_key].astype(int)
    person_ids = pd.unique(paths[id_key])

    ''' Split dataset according to training mode '''
    label_nums = dataset.label_nums_unified
    attack_nums = dataset.attack_nums_unified

    if training_mode == 'all_attacks':
        ''' Train on all attacks, test on all attacks '''
        # note: train_ids, val_ids, test_ids -- everything as a list
        paths['label'] = paths['label_unif']  # 0..4
        # split subsets based on person ids
        # val_ids, test_id = np.random.choice(person_ids, size=2, replace=False)
        # train_ids = np.setdiff1d(person_ids, [val_ids, test_ids])

        # IDs 1, 8, 19 are missing in the provided dataset
        # TODO manual indexes for now [clean]
        train_ids = np.array([2, 3, 4, 5, 6, 7, 9, 10, 11])
        val_ids = np.array([12])
        test_ids = np.array([13, 14, 15, 16, 17, 18, 20, 21, 22, 23])

        # split train/val/test (based on person IDs)
        paths_train = paths[paths[id_key].isin(train_ids)]
        paths_val = paths[paths[id_key].isin(val_ids)]
        paths_test = paths[paths[id_key].isin(test_ids)]

        class_train = class_val = class_test = 'all'

    elif training_mode == 'one_attack':
        ''' Train and test on one attack type '''
        # validation 1 person, test 1 unseen person, train the rest

        paths['label'] = paths['label_bin']  # 0,1

        # attack-splitting
        if args.attack_test == -1:
            # random
            class_train = class_val = class_test = np.random.choice(attack_nums, size=None,
                                                                    replace=False)  # note: originally size=1-> LIST
        else:
            # specific attack
            class_test = class_val = class_train = args.attack_test

        # person-splitting
        val_ids, test_ids = np.random.choice(person_ids, size=2, replace=False)
        train_ids = np.setdiff1d(person_ids, [val_ids, test_ids])

        # split train/val/test (based on attack type and person IDs)
        paths_train = paths[paths['label_unif'].isin([bona_fide, class_train])
                            & paths[id_key].isin(train_ids)]
        paths_val = paths[paths['label_unif'].isin([bona_fide, class_val])
                          & paths[id_key].isin([val_ids])]
        paths_test = paths[paths['label_unif'].isin([bona_fide, class_test])
                           & paths[id_key].isin([test_ids])]

    elif training_mode == 'unseen_attack':
        ''' Train on all attacks except one, test on the unseen attack '''
        # IDs not separated -- OK

        # note: test set == validation set
        paths['label'] = paths['label_bin']  # 0,1

        # attack-splitting
        if args.attack_test == -1:
            class_test = np.random.choice(attack_nums, size=None, replace=False)  # note: None-> INT
        else:
            class_test = args.attack_test

        class_val = class_test
        class_train = np.setdiff1d(attack_nums, [class_test])

        # person-splitting
        train_ids = val_ids = test_ids = person_ids

        # split train/val/test (based on attack type and person IDs)
        paths_train = paths[paths['label_unif'].isin(
            [bona_fide, *class_train])
            # todo: check in comparison to one_attack: *class_train, whereas there is no * [func]
            # & paths_all[id_key].isin(train_ids)
        ]
        paths_test = paths[paths['label_unif'].isin([bona_fide, class_test])
            # & (paths_all[id_key].isin(test_ids))
        ]
        paths_val = paths_test  # note: validation == test
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')

    ''' Safety check '''
    unique_classes = pd.concat([paths_train, paths_val, paths_test])['label'].nunique()
    if not unique_classes == num_classes:
        # will scream for RoseYoutu (no `other` class), but that's OK
        logging.warning(f'Number of unique classes in dataset does not match number of classes in model\n' +
                        f'real: {unique_classes}, expected: {num_classes}\n')

    assert bona_fide not in [class_test, class_val, *class_train], \
        'bona_fide label used as an attack label'
    # TODO check for string x int comparisons [warn]

    training_mode_long = training_mode
    if training_mode in ['unseen_attack', 'one_attack']:
        training_mode_long += f'_{class_test}'

    if True:
        ''' Set up directories and filenames '''
        dataset_lists_dir = 'dataset_lists'
        if not os.path.isdir(dataset_lists_dir) and not args.no_log:
            logging.info(f'Creating dir: {dataset_lists_dir}')
            os.makedirs(dataset_lists_dir)

        logging.info(f'Writing to dir: {dataset_lists_dir}')
        save_path_train = join(dataset_lists_dir, f'dataset_{dataset.name}_train_{training_mode_long}.csv')
        save_path_val = join(dataset_lists_dir, f'dataset_{dataset.name}_val_{training_mode_long}.csv')
        save_path_test = join(dataset_lists_dir, f'dataset_{dataset.name}_test_{training_mode_long}.csv')

        # quit if files already exist
        if isfile(save_path_train) or isfile(save_path_val) or isfile(save_path_test):
            logging.error('Annotation files already exist, quitting.')
            logging.error(f'Dataset: {dataset.name}, Training mode: {training_mode_long}')
            logging.error(f'Files: {save_path_train}, .. val, .. test')
            raise FileExistsError(
                f'Annotation files already exist, quitting. Dataset: {dataset.name}, Training mode: {training_mode_long}')

        else:  # create new files
            logging.info('Creating new annotation files.')
            logging.info(f'Dataset: {dataset.name}, Training mode: {training_mode_long}')
            logging.info(f'Files: {save_path_train}, .. val, .. test')

    ''' Shuffle, limit length '''
    # shuffling is done during dataset loading, not creation
    if False:
        # shuffle order - useful when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1, random_state=seed).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1, random_state=seed).reset_index(drop=True)

        # limit size for prototyping
        limit = -1  # -1 for no limit, 3200
        if limit != -1:
            note += f'LIMIT={limit}_'
            logging.warning(f'Limiting dataset (each split) to {limit} samples')

            paths_train = paths_train[:limit]
            paths_val = paths_val[:limit]
            paths_test = paths_test[:limit]

        logging.info('Dataset labels per split:')
        for p in [paths_train, paths_val, paths_test]:
            logging.info(p['label'].value_counts())

    save_path_metadata = join(dataset_lists_dir, f'dataset_{dataset.name}_metadata_{training_mode_long}.json')

    # metadata
    metadata = {
        'dataset_name': dataset.name,
        'training_mode': training_mode,
        'note': note,
        'num_classes': num_classes,
        'label_names': label_names,
        # attack splitting
        'attack_train': class_train,
        'attack_test': class_test,
        'attack_val': class_val,
        # person IDs
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        # split lengths
        'len_train': len(paths_train),
        'len_val': len(paths_val),
        'len_test': len(paths_test),
        # paths to annotation files
        'path_train': save_path_train,
        'path_val': save_path_val,
        'path_test': save_path_test,
        'path_metadata': save_path_metadata,
        'date_created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # print dictionary metadata
    metadata_list = ['Metadata:']
    for key, value in metadata.items():
        metadata_list.append(f'{key}: {value}')
    logging.info('\n'.join(metadata_list))

    ''' Save '''
    if not args.no_log:
        paths_train.to_csv(save_path_train, index=False)
        paths_val.to_csv(save_path_val, index=False)
        paths_test.to_csv(save_path_test, index=False)
        print('Saved dataset to:', save_path_train, save_path_val, save_path_test)
        ''' Save metadata '''
        save_dict_json(metadata, save_path_metadata)
        print('Saved metadata to:', save_path_metadata)
    else:
        logging.warning(f'Not saving dataset ({save_path_train}))')
        logging.warning(f'Not saving metadata ({save_path_metadata})')
        # print summary of dataset

    print('Dataset examples:')  # first 2 samples
    for name, p in zip(['train', 'val', 'test'], [paths_train, paths_val, paths_test]):
        print(name, '\n', p.head(2))

    print('Metadata:\n', metadata)

    # make csv out of metadata
    path_datasets_csv = join(dataset_lists_dir, 'datasets.csv')

    # make metadata dataframe-serializable (list to str)
    for key in metadata:
        if isinstance(metadata[key], np.ndarray):
            metadata[key] = metadata[key].tolist()
        if isinstance(metadata[key], list):  # no elif intentional
            metadata[key] = str(metadata[key])

    # convert metadata to dataframe, one row only, dictionary keys as columns
    df = pd.DataFrame(metadata, index=[0])

    if not args.no_log:
        # append to csv
        if not isfile(path_datasets_csv):
            df.to_csv(path_datasets_csv)
        else:
            df.to_csv(path_datasets_csv, mode='a', header=False)

    print('Datasets entry:\n', df)
