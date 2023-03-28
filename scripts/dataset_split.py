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
# following (3) arguments are for one_attack mode only
parser.add_argument('-t', '--attack_test', help='attack type to test on (1..7), random by default', type=int,
                    default=-1)
parser.add_argument('-v', '--attack_val', help='attack type to validate on (1..7), random by default', type=int,
                    default=-1)
parser.add_argument('-r', '--attack_train', help='attack type to train on (1..7), random by default', type=int,
                    default=-1)
parser.add_argument('-m', '--mode', help='unseen_attack, one_attack, all_attacks (see Readme)', type=str, required=True)

''' Create a dataset split (train, val, test) for a training mode (all_attacks, one_attack, unseen_attack) '''
'''
RoseYoutu contains 10 people. In every training mode:
Split into 8 for training, 1 for validation, 1 for testing
Every person has the same number of samples, but not the same number of attack types

Tested on:
                |  rose_youtu  |    siwm    |
all_attacks     | training OK  |    .       |
unseen_attack   |       .      |    .       |
one_attack      |       .      |    .       |

# generation failed for one_attack on metacentrum
'''

if __name__ == '__main__':
    """
    Create dataset training/val/test splits.
    """

    # set fixed seed
    seed = 42
    np.random.seed(seed)

    note = ''  # arbitrary extra info: dataset length limit, ...

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    training_mode = args.mode

    if args.dataset == 'rose_youtu':
        dataset = dataset_rose_youtu
    elif args.dataset == 'siwm':
        dataset = dataset_siwm
    else:
        raise ValueError('dataset must be in {rose_youtu, siwm}')

    bona_fide = 0  # todo does not apply for siwm, also read this from dataset

    if args.mode == 'one_attack' and xor(args.attack_test == -1, args.attack_train == -1):
        # todo limiting attacks (random x selected) not necessary, also include attack_val
        raise ValueError('one_attack mode requires both or none of --attack_test --attack_train arguments')

    assert bona_fide not in [args.attack_test, args.attack_train, args.attack_val], \
        'bona_fide label used as an attack label'

    # paths
    dataset_lists_dir = 'dataset_lists'
    if not os.path.isdir(dataset_lists_dir):
        logging.info(f'Creating dir: {dataset_lists_dir}')
        os.makedirs(dataset_lists_dir)

    logging.info(f'Writing to dir: {dataset_lists_dir}')
    save_path_train = join(dataset_lists_dir, f'dataset_{dataset.name}_train_{training_mode}.csv')
    save_path_val = join(dataset_lists_dir, f'dataset_{dataset.name}_val_{training_mode}.csv')
    save_path_test = join(dataset_lists_dir, f'dataset_{dataset.name}_test_{training_mode}.csv')

    # quit if files already exist
    if isfile(save_path_train) or isfile(save_path_val) or isfile(save_path_test):
        logging.error('Annotation files already exist, quitting.')
        logging.error(f'Dataset: {dataset.name}, Training mode: {training_mode}')
        logging.error(f'Files: {save_path_train}, .. val, .. test')
        raise FileExistsError(
            f'Annotation files already exist, quitting. Dataset: {dataset.name}, Training mode: {training_mode}')

    else:  # create new files
        logging.info('Creating new annotation files.')
        logging.info(f'Dataset: {dataset.name}, Training mode: {training_mode}')

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

    # get annotations (paths to samples + labels)
    paths_genuine, paths_attacks = dataset.read_annotations('both')
    paths_all = pd.concat([paths_genuine, paths_attacks])

    person_ids = pd.unique(paths_all['id0'])

    ''' Split dataset according to training mode '''
    label_nums = dataset.label_nums_unified
    attack_nums = dataset.attack_nums_unified

    if training_mode == 'all_attacks':
        ''' Train on all attacks, test on all attacks '''

        paths_all['label'] = paths_all['label_unif']  # 0..4
        # split subsets based on person ids
        val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
        train_ids = np.setdiff1d(person_ids, [val_id, test_id])

        # split train/val/test (based on person IDs)
        paths_train = paths_all[paths_all['id0'].isin(train_ids)]
        paths_val = paths_all[paths_all['id0'].isin([val_id])]
        paths_test = paths_all[paths_all['id0'].isin([test_id])]

        class_train = class_val = class_test = 'all'

    elif training_mode == 'one_attack':
        ''' Train and test on one attack type '''

        # train on all persons, test on one unseen person

        paths_all['label'] = paths_all['label_bin']  # 0,1

        # attack-splitting
        if args.attack_test == -1:
            # random
            class_train = class_val = class_test = np.random.choice(attack_nums, size=1, replace=False)
        else:
            # specific attack
            class_train = args.attack_train
            class_test = args.attack_test
            # random other class for validation
            # class_val = np.random.choice(np.setdiff1d(attack_nums, [class_train, class_test]), size=None)
            class_val = args.attack_val

        # person-splitting
        val_id, test_id = np.random.choice(person_ids, size=2, replace=False)
        train_ids = np.setdiff1d(person_ids, [val_id, test_id])

        # split train/val/test (based on attack type and person IDs)
        paths_train = paths_all[paths_all['label_unif'].isin([bona_fide, class_train])
                                & paths_all['id0'].isin(train_ids)]
        paths_val = paths_all[paths_all['label_unif'].isin([bona_fide, class_val])
                              & paths_all['id0'].isin([val_id])]
        paths_test = paths_all[paths_all['label_unif'].isin([bona_fide, class_test])
                               & paths_all['id0'].isin([test_id])]

    elif training_mode == 'unseen_attack':
        ''' Train on all attacks except one, test on the unseen attack '''

        # note: test set == validation set
        paths_all['label'] = paths_all['label_bin']  # 0,1

        # attack-splitting
        class_test = np.random.choice(attack_nums, size=None, replace=False)
        class_val = class_test
        class_train = np.setdiff1d(attack_nums, [class_test])

        # person-splitting
        test_id = np.random.choice(person_ids, size=None, replace=False)
        val_id = test_id
        train_ids = np.setdiff1d(person_ids, [test_id])

        # split train/val/test (based on attack type and person IDs)
        paths_train = paths_all[paths_all['label_unif'].isin(
            [bona_fide, *class_train])  # todo: check in comparison to one_attack: *class_train, whereas there is no *
                                & paths_all['id0'].isin(train_ids)]
        paths_test = paths_all[paths_all['label_unif'].isin([bona_fide, class_test])
                               & (paths_all['id0'] == test_id)]
        paths_val = paths_test  # note: validation == test
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')

    ''' Safety check '''
    unique_classes = pd.concat([paths_train, paths_val, paths_test])['label'].nunique()
    if not unique_classes == num_classes:
        logging.warning(f'Number of unique classes in dataset does not match number of classes in model\n' +
                        f'real: {unique_classes}, expected: {num_classes}\n')

    training_mode_long = training_mode
    if training_mode in ['unseen_attack', 'one_attack']:
        training_mode_long += f'_{class_test}'

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
        'val_id': val_id,
        'test_id': test_id,

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

    ''' Shuffle, limit length '''
    if False:
        # shuffle order - useful when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1, random_state=seed).reset_index(
            drop=True)  # todo random_state for reproducibility
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
        for paths in [paths_train, paths_val, paths_test]:
            logging.info(paths['label'].value_counts())

    # print dictionary metadata
    metadata_list = ['Metadata:']
    for key, value in metadata.items():
        metadata_list.append(f'{key}: {value}')
    logging.info('\n'.join(metadata_list))

    ''' Save '''
    paths_train.to_csv(save_path_train, index=False)
    paths_val.to_csv(save_path_val, index=False)
    paths_test.to_csv(save_path_test, index=False)

    ''' Save metadata '''
    save_dict_json(metadata, save_path_metadata)

    # make csv out of metadata
    path_datasets_csv = join(dataset_lists_dir, 'datasets.csv')

    # make metadata dataframe-serializable (list to str)
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()
        if isinstance(value, list):  # no elif intentional
            metadata[key] = str(value)

    df = pd.DataFrame(metadata, index=[0])

    # append to csv
    if not isfile(path_datasets_csv):
        df.to_csv(path_datasets_csv)
    else:
        df.to_csv(path_datasets_csv, mode='a', header=False)
