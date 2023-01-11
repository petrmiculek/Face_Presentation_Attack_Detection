# stdlib
# -

# external
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ConvertImageDtype, ToPILImage

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
