# run default training script

# detect if running on local
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Running on local machine"
    export CUDA_VISIBLE_DEVICES=0
    py="python3"
    batch_size=1
    num_workers=4
    epochs=1
    seed=42
else
    echo "Running on cluster"
    py=/opt/conda/bin/python3
    batch_size=16
    num_workers=8
    epochs=20
    seed=$RANDOM
fi

# run training

# default args:
# -d rose_youtu
# -m all_attacks

$py src/train.py -a efficientnet_v2_s -l 0.0001 -b $batch_size -e $epochs -w $num_workers -s "$seed"

# run from root repo directory
