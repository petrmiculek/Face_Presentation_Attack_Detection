doing:

* seed:
  * apply for instantiating the model (weights) - verify same weights when not loading pretrained
  * TypeError: DataLoader._\_init_\_() got an unexpected keyword argument 'seed' #DONE#
  * delete printing seed after verifying it works (from dataset_base.py)

* compare explanations of two different models

* verify training runs after training loop simplification #DONE#
* verify evaluation runs with unified load_model #DONE#

* do transforms in DataLoader #DONE#

* lime:
  * just call it #DONE#
  * generate many
  *
* many-run script can create run_x.sh files + finished files

############################## HOT

* input size of EfficientNet_V2S is 384x384, whereas Resnet has 224x224
  * transforms=partial(
    ImageClassification,
    crop_size=384,
    resize_size=384,
    interpolation=InterpolationMode.BILINEAR,
    )
  * things work fine, it's just that the resizing might be doing some unnecessary work

############################## HOT

* pred_hwc_np uses F.softmax - is it correct? Why is it not used in the training? What comes out of the model?
* It should be easy to make EfficientNet produce its one-to-last layer embeddings.

* rose_youtu has 'Genuine', siw-m has 'Live'
* siw-m
  * has negative category as 1 -> harder for binary labels
  * has no 'label_bin' attribute
  *

todo:

* train with SIW-M dataset
* dataset-independent train: test
* saving model: make sure it can be re-initialized (seems to work for evaluate)
* metriky: EER = equal error rate, HTER, AUC, XYZ@FRR10e-2
* report real number of epochs trained + training time
* return samples as dict to enable extending annotations to paths, identities, etc.
* training: one_attack, unseen_attack
* extract one-to-last layer embeddings (for t-SNE etc.)
* log the plots to wandb
* check one-attack splitting in dataset_split (fill in "OK" table in dataset_split.py)
* regenerate one-attack datasets, with new attack splitting
* one_attack training:
* script: run training on every category separately TODO
* possibly include out-of-distribution data for testing? #REJECT#
* fixed seed for train.py
* preprocess(sample) on cpu or gpu?
* prep.sh - read job number somewhere from the environment

unseen_attack training:

* script: run for every category as unseen

less important:

* fix W&B 'failed to sample metrics' error
* checkpoint also state of scheduler, optimizer, ... https://docs.wandb.ai/guides/track/advanced/resuming
  * ^but separately from model weights, to load independently
* reuse code for eval_loop
* default parameters for model architecture hide intent

ideas:

* train for longer to check if val will diverge
* if all_attacks, report the binary metrics (or always binary and for some modes it will be the same as "non-binary")
* follow the evaluation protocol of SIW-M
* W\&B profiling
* re-name categories to use only main ones (mask, printed, video, etc.) (adapt RoseYoutu to SIW-M)

* rethinking classes:
  'Makeup',
  'Live',
  'Paper',
  'Mask_partial',
  'Replay',
  'Mask'
  ->
  'Genuine',
  'Printed',
  'Video',
  'Mask',
  'Other'
  * adapting rose_youtu #DONE#
    * train again on metacentrum #DONE#
  * adapt siw-m

note:

* val == test for unseen_attack
* no bbox faces from siw-m excluded from training, could be used for extra human eval
* dataset was cropped by resizing the bbox to a square (as opposed to keeping the original aspect ratio)
* you're also explaining the code - mention the use-cases and give short script descriptions
* slowing down training: images in original size, many small files (h5)
* 2023-03-21: changed 'label_num' to 'label_unif', all newer models are not comparable

#DONE# #DONE# #DONE#
done:

* save config to json #DONE#
* read SIW-M dataset #DONE#
* random dataset creation for every run hurts correct/valid/repeatable evaluation and verification later #DONE#
* splitting datasets (label_num, id0, ... ) -- maybe do this within the dataset.py itself? #DONE#
* => generate the dataset split only once, export it (separate script?)  #DONE#
* one file with paths to datasets (=> dataset_xyz.py then knows the structure, but not the root path?)  #DONE#
* I can generate all \*-cam outputs #DONE#
* I have 0 control over dataset split for all_attacks - person_ids are random and not saved (now saved)  #DONE#
* dataset size is being limited to 640 (not anymore) #DONE#
* efficient_net training works, but we don't init the model with 8 classes #DONE#

* setup for metacentrum #DONE#
* 16bit training #DONE#
* gpu training #DONE#
* resnet18.py * local model implementation #DONE#
* eval metrics #DONE#
* validation dataset split #DONE#
* log the class chosen for training/val/test #DONE#
* include the class names in the confusion matrix title #DONE#
* W&B #DONE#
* confusion matrix #DONE#
  one_attack training:
  * train genuine + one type of attack #DONE#
  * binary predictions #DONE#
  * shuffling the dataset #DONE#
  * mixing genuine and attack data #DONE#
    unseen_attack training:
  * train genuine + 6/7 categories #DONE#
  * test on the last category #DONE#
* print confmat full pandas dataframe (don't skip columns)  #DONE#
* check model name saved -> load model in evaluate #DONE#
* cache function calls (reading annotations?)  #SKIPPED#
* I was evaluating the binary accuracy of the model #DONE#
  * check if multi-class accuracy is reported now, and fed to W\&B #DONE#
* Confusion matrix
  * check that closing the figure fixes the text overwriting with multi-class and binary #DONE#

#DONE# #DONE# #DONE#

inspiration:

* CelebA spoof:
  * (paper)
    * evaluation metrics table
    * schema of model inputs and outputs

  * (code)
    * implementation of metrics: FRR@10-3 (intra_dataset_code/client.py)

consultation:

* why is liveness solved by a separate model at the age of massive universal backbone, multitask networks, transfer
  learning/fine-tuning successes?
* metacentrum - have you tried keeping a dataset in-memory? it's like flying

low-prio:

* union of dicts - is it order-dependent? key collisions
* running dataset_split.py from the scripts folder will fail to import src.\*, handle that in the .sh script

problems:

* dataset_split for one_attack did not work on metacentrum -- 1 class found, 2 expected
* pipenv might not be available on metacentrum // not necessary with currently used packages (grad-cam, lime not tested)
* confusion matrix pdf binary has 2 legends and overwritten text #DONE#
* some earlier runs (astral-paper-14)' confusion matrix counts don't add up, ok on eval

write:

* implementation - training done on metacentrum, dataset kept in memory (shm), 16-bit training
* reproducibility levels - dataset split, model initialization, training parameters, batch_size => drop_last

#############
trash bin:

1)

one_attack: figure out splitting
persons w.r.t. attack and genuine
a) train on person 1, test on person 2
b) train on all persons, test on one unseen person #CHOSEN#


