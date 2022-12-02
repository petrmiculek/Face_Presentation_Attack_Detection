training_run_id = None  # initialized from train.py

HPARAMS = {
    'epochs': 1,
    'batch_size': 32,
    'lr': 0.001,
    'early_stopping_patience': 5,
    'weight_decay': 1e-3,

    'lr_scheduler_patience': 3,
    'lr_scheduler_min_lr': 1e-6,
    'lr_scheduler_factor': 0.5,
}
