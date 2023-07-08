# stdlib
from os.path import join, exists

# external
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage
import pandas as pd
import torch
import random
import numpy as np


# local
# -

class BaseDataset(Dataset):
    path_key = 'path_key'  # todo unused [clean]

    def __init__(self, annotations, transform=None):
        self.transform = transform
        self.samples = annotations

        # test first file exists
        sample = self.samples.iloc[0]
        path = sample['path']
        if not exists(path):
            raise FileNotFoundError(f'Sample dataset file does not exist: {path} ')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        image = Image.open(sample['path'])
        if self.transform:
            image = self.transform(image)

        sample_dict = {
            'image': image,
            'label': sample['label'],
            'path': sample['path'],
            'idx': sample['idx']
        }

        return sample_dict


def StandardLoader(dataset_class, annotations, **kwargs):
    """
    Returns a dataloader for the Rose Youtu dataset
    :param dataset_class: Dataset class to use
    :param annotations: DataFrame with list of files
    :param kwargs: keyword arguments for DataLoader
    :return: dataloader

    possibly add:
    - fraction to limit dataset size
    - random seed + fixed  #DONE#
    - ddp
    - drop_last  #DONE#
    -
    """

    shuffle = kwargs.pop('shuffle', False)
    batch_size = kwargs.pop('batch_size', 1)
    num_workers = kwargs.pop('num_workers', 1)
    seed = kwargs.pop('seed', None)
    transform = kwargs.pop('transform', None)
    drop_last = kwargs.pop('drop_last', True)

    def seed_worker(worker_id):
        """
        Seed worker for dataloader.
        The seed set in main process is propagated to all workers,
        but only within Pytorch - not all other libraries.

        :param worker_id:
        """
        if seed is not None:
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    kwargs_dataset = {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'pin_memory': True,
        'drop_last': drop_last,
        'worker_init_fn': seed_worker,
        'shuffle': shuffle,
    }

    # this did not work
    # if seed is not None:
    #     g = torch.Generator()
    #     g.manual_seed(seed)
    #     kwargs_dataset['generator'] = g

    kwargs_dataset.update(kwargs)

    if transform is None:
        transform = Compose([
            # Resize((224, 224)),
            ToTensor(),  # transforms.PILToTensor(),
            # ConvertImageDtype(torch.float),
            # Normalize(mean=[0.485, 0.456, 0.406],
        ])

    dataset = dataset_class(annotations, transform=transform)
    loader = DataLoader(dataset, **kwargs_dataset)

    return loader


def pick_dataset_version(name, mode, attack=None):
    """
    Pick dataset version.

    note: `available` is updated, as `datasets` is filtered.
    ^ Don't try to simplify the code by reusing `available` many times.

    :param name: dataset name
    :param mode: training mode
    :param attack: attack number
    :return: metadata pandas series
    """
    path_datasets_csv = join('dataset_lists', 'datasets.csv')  # todo make into a parameter [clean]
    datasets = pd.read_csv(path_datasets_csv)
    available = datasets[['dataset_name', 'training_mode']].values

    # filter by name
    datasets = datasets[datasets['dataset_name'] == name]
    if len(datasets) == 0:
        raise ValueError(f'No dataset with name {name}. '
                         f'Available datasets:\n{available}')

    available = datasets[['dataset_name', 'training_mode']].values
    # filter by training mode
    datasets = datasets[datasets['training_mode'] == mode]
    if len(datasets) == 0:
        raise ValueError(f'No dataset with name {name} and mode {mode}. '
                         f'Available datasets:\n{available}')

    if attack is not None and mode == 'all_attacks':
        print(f'Ignoring attack number, training mode is: {mode}')
        attack = None

    available = datasets[['dataset_name', 'training_mode', 'attack_test']].values
    # filter by attack
    if attack is not None:
        datasets = datasets[datasets['attack_test'] == attack]
        if len(datasets) == 0:
            raise ValueError(f'No dataset with name {name}, mode {mode} and attack {attack}. '
                             f'Available datasets:\n{available}')

    elif len(datasets) > 1:
        available = datasets[['dataset_name', 'training_mode', 'attack_test']].values
        raise ValueError(f'Multiple datasets with name {name} and mode {mode}. '
                         f'Available datasets:\n{available}')

    return datasets.iloc[0]


def load_annotations(metadata_row, seed, limit=-1, shuffle=False, quiet=True):
    # load annotations
    paths_train = pd.read_csv(metadata_row['path_train'])
    paths_val = pd.read_csv(metadata_row['path_val'])
    paths_test = pd.read_csv(metadata_row['path_test'])
    # add index column "idx"
    paths_train['idx'] = paths_train.index
    paths_val['idx'] = paths_val.index
    paths_test['idx'] = paths_test.index
    # shuffle initial order
    if limit != -1 or shuffle:
        # shuffle when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1, random_state=seed).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1, random_state=seed).reset_index(drop=True)

        # limit dataset size
        if not quiet:
            print(f'Limiting dataset (each split) to {limit} samples.')
        paths_train = paths_train[:limit]
        paths_val = paths_val[:limit]
        paths_test = paths_test[:limit]

    return {'train': paths_train, 'val': paths_val, 'test': paths_test}


def load_dataset(metadata_row, dataset_module, limit=-1, quiet=False, **loader_kwargs):
    """
    Load dataset from metadata.

    :param metadata_row: paths to files and metadata
    :param dataset_module: python module with dataset class (see note)
    :param limit: limit dataset size (enables shuffling)
    :param quiet: silent mode
    :param loader_kwargs: keyword arguments for dataset loader (see below)

    loaders_kwargs:
        - seed:
        - shuffle:
            shuffle datasets during loading (list of their files -> same for all epochs),
            also shuffles training dataset inside DataLoader (on each epoch)

    :return: dataset loaders for train, val, test

    :note: dataset_module used so that we don't import individual datasets here -> cycle
    """
    # name = metadata_row['dataset_name']  # todo could reimport dataset module here [clean]
    seed = loader_kwargs.pop('seed', None)
    if 'transform' in loader_kwargs:
        transform_train = transform_eval = loader_kwargs.pop('transform')
    else:
        transform_train = loader_kwargs.pop('transform_train', None)
        transform_eval = loader_kwargs.pop('transform_eval', None)

    shuffle = loader_kwargs.pop('shuffle', False)
    paths = load_annotations(metadata_row, seed, limit, shuffle, quiet)

    ''' print label distributions '''
    if not quiet:
        num_classes = int(metadata_row['num_classes'])
        print('Dataset labels per split:')  # including labels not present
        # it = zip(['train', 'val', 'test'], [paths_train, paths_val, paths_test])
        for split, split_paths in paths.items():
            # TODO call show_labels_distribution [clean]
            class_occurences = []
            value_counts = split_paths['label'].value_counts().sort_index()
            for i in range(num_classes):
                if i in value_counts.index:
                    class_occurences.append(value_counts[i])
                else:
                    class_occurences.append(0)

            print(f'{split}:', class_occurences)

    ''' Data loaders '''
    loader_train = dataset_module.Loader(paths['train'], seed=seed, transform=transform_train, **loader_kwargs,
                                         shuffle=shuffle)
    loader_val = dataset_module.Loader(paths['val'], seed=seed, transform=transform_eval, **loader_kwargs)
    loader_test = dataset_module.Loader(paths['test'], seed=seed, transform=transform_eval, **loader_kwargs)

    return loader_train, loader_val, loader_test


def show_labels_distribution(labels, split, num_classes):
    """
    Print labels distribution for a split.

    :param labels: Pandas Series with labels
    :param split: String with split name
    :param num_classes: Number of classes
    """
    value_counts = labels.value_counts().sort_index()
    class_occurences = []
    for i in range(num_classes):
        if i in value_counts.index:
            class_occurences.append(value_counts[i])
        else:
            class_occurences.append(0)
    print(f'{split}:', class_occurences)


def get_dataset_setup(dataset_module, training_mode):
    """Get dataset setup (label names, number of classes) for a given training mode."""
    label_names_binary = ['genuine', 'attack']
    if training_mode == 'all_attacks':
        label_names = dataset_module.label_names_unified
        num_classes = len(dataset_module.labels_unified)
    elif training_mode == 'one_attack':
        label_names = label_names_binary
        num_classes = 2
    elif training_mode == 'unseen_attack':
        label_names = label_names_binary
        num_classes = 2
    else:
        raise ValueError(f'Unknown training mode: {training_mode}')
    return label_names, label_names_binary, num_classes
