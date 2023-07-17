training_run_id = None  # initialized from train.py

# default training hyperparameters
HPARAMS = {
    'epochs': 1,
    'batch_size': 2,
    'lr': 1e-4,
    'early_stopping_patience': 5,
    'weight_decay': 1e-3,

    'lr_scheduler_patience': 3,
    'lr_scheduler_min_lr': 1e-6,
    'lr_scheduler_factor': 0.5,
}

seed_eval_default = 42

sample_shape = (1, 3, 384, 384)

# -----------------------
# Explanations Evaluation
# blurring CAM mask
blur_cam_s = int(384 // 2) + 1  # == 193; 384 is the size of the input image; = sigma in blur; odd number
# blurring input image
blur_img_s = 29  # empirically found to drop prediction confidence below chance level; odd number

# -----------------------
# Paths
# -----------------------
from os.path import join

dataset_lists_dir = 'dataset_lists'  # directory for dataset lists (annotations and paths)
path_datasets_csv = join(dataset_lists_dir, 'datasets.csv')  # list of datasets
runs_dir = 'runs'  # directory for saving models, logs, etc.
