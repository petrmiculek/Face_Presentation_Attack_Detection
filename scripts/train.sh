# run default training script

# detect if running on local
if [ -z "$SCRATCHDIR" ]; then
    echo "# Running on local machine"
#    export CUDA_VISIBLE_DEVICES=0
    py="python3"
    batch_size=1
    num_workers=4
    epochs=1
    seed=42
else
    echo "# Running on cluster"
    py=/opt/conda/bin/python3  # 3.8.X on singularity 22.10
#    py=/usr/bin/python  # 3.8.10 on singularity 23.02
    batch_size=16
    num_workers=16
    epochs=10
    seed=$RANDOM
fi

# run training

# default args:
# -d rose_youtu
# -m all_attacks

$py src/train.py -a efficientnet_v2_s -l 0.0001 -b $batch_size -e $epochs -w $num_workers -s "$seed"

# batch size:
# glados, RTX 2080, 8GB -> 16; leaked 5GB during the work :/, glados cannot run singularity23.02
# adan, 16GB
# in speed benchmark for empty training loop, batch size 64 is used - not realistic for training
