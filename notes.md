

todo: 
* train with SIW-M dataset
* dataset-independent train:
  * splitting datasets (label_num, id0, ... ) -- maybe do this within the dataset.py itself?
  *
* saving model: make sure it can be re-initialized



ideas:

* generate the dataset split only once, export it (separate script?)
* 

note:
23-01-10

* dataset size is being limited to 640 
* val == test for unseen_attack
* no bbox faces from siw-m, excluded from training, could be used for extra human eval
* dataset was cropped by resizing the bbox to a square (as opposed to keeping the original aspect ratio)
* you're also explaining the code - mention the use-cases and give short script descriptions

done:

* save config to json
* read SIW-M dataset


inspiration

* CelebA spoof: 
  * (paper)
    * evaluation metrics table
    * schema of model inputs and outputs
  
  * (code)
    * implementation of metrics: FRR@10-3 (intra_dataset_code/client.py)
