doing:

############################## HOT

############################## HOT

* rose_youtu has 'Genuine', siw-m has 'Live'
* siw-m
  * has negative category as 1 -> harder for binary labels
  * has no 'label_bin' attribute
  *

todo:

* how about lime?

* train with SIW-M dataset
* dataset-independent train: test
* saving model: make sure it can be re-initialized (seems to work for evaluate)
* metriky: EER = equal error rate, HTER, AUC, XYZ@FRR10e-2

ideas:

* if all_attacks, report the binary metrics (or always binary and for some modes it will be the same as "non-binary")
* follow the evaluation protocol of SIW-M

note:

* dataset size is being limited to 640
* val == test for unseen_attack
* no bbox faces from siw-m excluded from training, could be used for extra human eval
* dataset was cropped by resizing the bbox to a square (as opposed to keeping the original aspect ratio)
* you're also explaining the code - mention the use-cases and give short script descriptions
* slowing down training: images in original size, many small files (h5)

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
*

low-prio:

* union of dicts - is it order-dependent? key collisions
* running dataset_split.py from the scripts folder will fail to import src.*, handle that in the .sh script

problems:

* dataset_split for one_attack did not work on metacentrum -- 1 class found, 2 expected
* pipenv might not be available on metacentrum
*

#############
trash bin:

1)

one_attack: figure out splitting
persons w.r.t. attack and genuine
a) train on person 1, test on person 2
b) train on all persons, test on one unseen person #CHOSEN#
