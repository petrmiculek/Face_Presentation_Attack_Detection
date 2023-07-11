# Explainable Face Liveness Classification
= Face Presentation Attack Detection (PAD)
= Spoofing Detection

Master Thesis at the Faculty of Information Technology, Brno University of Technology

# Author
Petr Mičulek <petr.miculek@gmail.com>

[//]: # (## Abstract)

## Data

RoseYoutu Dataset

[//]: # (link to download, instructions)

## How to run

[//]: # (update Pipfile)
[//]: # Just run the prep.sh

```
# install requirements from Pipfile
pipenv install


[//]: # NOTE: `pip install lime` also works, so it's in the Pipfile
[//]: # (# install lime library from the repo)
[//]: # (https://github.com/marcotcr/lime/tree/master/lime)
[//]: # (pipenv shell)
[//]: # (# clone)
[//]: # (# cd to repo)
[//]: # (pip install .)

# activate environment
pipenv shell
python3.10 src/train.py --mode <mode>
```

[//]: # (TODO specify mode)

\-\- mode argument:
unseen_attack - train on all attacks except one, test on the remaining attack
one_attack - train on one attack, test on another attack
all_attacks - train on all attacks, test on all attacks (different people across subsets)


[//]: # (first run will download the model weights)

## Results

[//]: # (add results)

# TODO Add this to the top of each file
```python3
#! /usr/bin/env python3
__author__ = 'Petr Mičulek'
__project__ = 'Master Thesis - Explainable Face anti-spoofing'
__date__ = '31/07/2023'
# TODO Describe this file
"""
Dataset Rose-Youtu

"""
```


