# run default training script

# detect if running on local
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Running on local machine"
    export CUDA_VISIBLE_DEVICES=0
    python="python3"
    batch_size=1
    num_workers=4
    epochs=1
else
    echo "Running on cluster"
    python=/opt/conda/bin/python3
    batch_size=16
    num_workers=8
    epochs=20
fi

# run training

# default args:
# -d rose_youtu
# -m all_attacks

$python src/train.py -a efficientnet_v2_s -l 0.0001 -b $batch_size -e 1 -w $num_workers

# run from root repo directory
