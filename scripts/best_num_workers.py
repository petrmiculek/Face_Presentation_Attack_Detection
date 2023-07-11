import os
import sys

sys_path_extension = [os.getcwd(), os.path.join(os.pardir, 'src')] + \
                     [d for d in os.listdir() if os.path.isdir(d)] + \
                     [d for d in os.listdir(os.pardir) if os.path.isdir(d)]
sys.path.extend(sys_path_extension)
from time import perf_counter
import multiprocessing as mp
from tqdm import tqdm
from src.dataset_base import load_dataset, pick_dataset_version
import src.dataset_rose_youtu as dataset_module

""" Find optimal num_workers value for dataloader """

batch_size = 64
limit = -1
dataset_meta = pick_dataset_version('rose_youtu', 'all_attacks')

for num_workers in range(2, mp.cpu_count() + 1, 2):
    loader_kwargs = {'shuffle': True, 'batch_size': batch_size, 'num_workers': num_workers,
                     'pin_memory': True, 'seed': 1}
    path_prefix = ...  # TODO set path_prefix
    train_loader, val_loader, test_loader = load_dataset(dataset_meta, dataset_module, path_prefix=path_prefix,
                                                         limit=limit, quiet=True, **loader_kwargs)
    start = perf_counter()
    try:
        for epoch in range(1, 3):
            for i, data in tqdm(enumerate(train_loader, 0), desc=f'{num_workers=},{epoch=}', total=len(train_loader)):
                pass
        end = perf_counter()
        print(f'\t\t\tFinish with:{end - start:.4f} second, {num_workers=}')
    except KeyboardInterrupt as e:
        pass

"""
best results found:
memory-mapped dataset, 32 cpus metacentrum, 128gb ram
num_workers = 8
full rose_youtu dataset read in <30s
real training epoch in 1min
"""
