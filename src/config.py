training_run_id = None  # initialized from train.py

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

sample_shape = (1, 3, 224, 224)  # TODO hardcoded input size

# -----------------------
# Explanations Evaluation
# blurring CAM mask
blur_cam_s = int(386 // 2)  # == 193; 386 is the size of the input image; = sigma in blur; odd number
# blurring input image
blur_img_s = 29  # empirically found to drop prediction confidence below chance level; odd number
