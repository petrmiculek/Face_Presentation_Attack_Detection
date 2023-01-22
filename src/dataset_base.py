# stdlib
from os.path import join

# external
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage
import pandas as pd

# local
# -

class BaseDataset(Dataset):
    path_key = 'path_key'

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

def StandardLoader(dataset_class, annotations, **kwargs):
    """
    Returns a dataloader for the Rose Youtu dataset
    :param dataset_class: Dataset class to use
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

    dataset = dataset_class(annotations, transform=transform)
    loader = DataLoader(dataset, **kwargs_dataset)

    return loader


def pick_dataset_version(name, mode):
    """
    Pick dataset version.

    :param name: dataset name
    :param mode: training mode
    :return: metadata pandas series
    """
    path_datasets_csv = join('config', 'datasets.csv')
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

    elif len(datasets) > 1:
        available = datasets[['dataset_name', 'training_mode']].values
        raise ValueError(f'Multiple datasets with name {name} and mode {mode}. '
                         f'Available datasets:\n{available}')

    return datasets.iloc[0]


def load_dataset(metadata_row, dataset_module, limit=-1, quiet=True, **loader_kwargs):
    """
    Load dataset from metadata.

    :param metadata_row: paths to files and metadata
    :param dataset_module: python module with dataset class (see note)
    :param limit: limit dataset size (enables shuffling)
    :param quiet: silent mode
    :param loader_kwargs: keyword arguments for dataset loader
    :return: dataset loaders for train, val, test

    :note: dataset_module used so that we don't import individual datasets here -> cycle
    """
    # name = metadata_row['dataset_name']  # could reimport dataset module here

    # load annotations
    paths_train = pd.read_csv(metadata_row['path_train'])
    paths_val = pd.read_csv(metadata_row['path_val'])
    paths_test = pd.read_csv(metadata_row['path_test'])

    shuffle = loader_kwargs.pop('shuffle', False)
    # shuffle initial order
    if limit != -1 or shuffle:
        # shuffle when limiting dataset size to keep classes balanced
        paths_train = paths_train.sample(frac=1).reset_index(drop=True)
        paths_val = paths_val.sample(frac=1).reset_index(drop=True)
        paths_test = paths_test.sample(frac=1).reset_index(drop=True)

        # limit dataset size
        print(f'Limiting dataset (each split) to {limit} samples.')
        paths_train = paths_train[:limit]
        paths_val = paths_val[:limit]
        paths_test = paths_test[:limit]

    # print label distributions
    if not quiet:
        print('Dataset labels per split:')
        it = zip(['train', 'val', 'test'], [paths_train, paths_val, paths_test])
        for split, paths in it:
            print(f'{split}:', list(paths['label'].value_counts().sort_index()))

    # data loaders
    loader_train = dataset_module.Loader(paths_train, **loader_kwargs, shuffle=shuffle)
    loader_val = dataset_module.Loader(paths_val, **loader_kwargs)
    loader_test = dataset_module.Loader(paths_test, **loader_kwargs)

    return loader_train, loader_val, loader_test
