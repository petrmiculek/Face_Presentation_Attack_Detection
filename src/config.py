#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'

"""
Configuration file for the project.

constants, paths, and hyperparameters
to be accessed from anywhere.
"""
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

# image size - native, used for efficientnet, resnet uses (224, 224)
sample_shape = (1, 3, 384, 384)

# -----------------------
# Explanations Evaluation
# blurring CAM mask
blur_cam_s = int(384 // 2) + 1  # == 193; 384 is the size of the input image; = sigma in blur; odd number
# blurring input image
blur_img_s = 29  # empirically found to drop prediction confidence below chance level; odd number
blur_mask_s = 5  # prevent sharp edges in the mask

cam_blurred_weight = 0.5  # weight of the blurred CAM mask in the final mask

# Deletion Metric - percentage of pixels kept in the mask
percentages_kept = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]  # before: [100, 90, 70, 50, 30, 10, 0]

# -----------------------
# Paths
# -----------------------
from os.path import join

dataset_lists_dir = 'dataset_lists'  # directory for dataset lists (annotations and paths)
path_datasets_metadata = join(dataset_lists_dir, 'datasets.pkl')  # list of datasets
runs_dir = 'runs'  # directory for saving models, logs, etc.
cam_dir_name = 'cam'
lime_dir_name = 'lime'
