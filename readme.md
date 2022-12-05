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
```
# install requirements from Pipfile
pipenv install

pipenv shell
python3.10 src/train.py --mode
```

\-\- mode argument:
unseen_attack - train on all attacks except one, test on the remaining attack
one_attack - train on one attack, test on another attack
all_attacks - train on all attacks, test on all attacks (different people across subsets)

## Results

[//]: # (add results)



