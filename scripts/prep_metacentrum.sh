#! /usr/bin/bash
# qsub -I -l select=1:ncpus=4:mem=32gb:ngpus=1:scratch_shm=true -q gpu -l walltime=1:30:00

# copy files to scratch dir (memory-mapped)
cd s_brno2 || echo "ERROR: cd failed"
cp -r data.tar "$SCRATCHDIR"
tar -xf "$SCRATCHDIR"/data.tar --directory "$SCRATCHDIR"/
# cp -r config config_backup
# path normalls starts with $FIND
cd s_brno2/facepad || echo "ERROR: cd failed"
if [ -z "$SCRATCHDIR" ]; then
  echo "# Running on local machine"
  export FIND='../data/'
  export REPLACE='/dev/shm/'
else
  echo "# Running on cluster"
  # read header, count number of commas before path column
  header_idx_path="$(head -n 1 dataset_lists/dataset_rose_youtu_test_all_attacks.csv | tr ',' '\n' | grep -n 'path' | cut -d ':' -f 1)"
  #  export FIND='..'  # if dataset never overwritten
  first_lines="$(head dataset_lists/dataset_*.csv -n 3 | tail -n 1)"

  export FIND="${first_lines%%rose_youtu_crops4*}"  # TODO temporary
  export REPLACE="$SCRATCHDIR/rose_youtu_full/"
#  export REPLACE="/storage/brno2/home/petrmiculek/rose_youtu_full/"
fi

echo "# FIND=$FIND"
echo "# REPLACE=$REPLACE"

# edit copies of dataset annotation files
tmp_counter=1
for file in dataset_lists/dataset_*.csv; do
	out_tmp="dataset_lists/tmp_$tmp_counter.csv"
	echo "perl -pe 's|\Q$FIND\E|$REPLACE|g' $file>$out_tmp"
	echo "mv $out_tmp $file"
	tmp_counter=$((tmp_counter+1))
done

# unused
#for file in dataset_lists/dataset_*; do
#	echo "perl -pe 's|\Q$FIND\E|$SCRATCHDIR|g' $file">
#done

#singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.05-py3.SIF
# problem: torchvision efficientnet_v2_s: unexpected keyword argument 'weight_class'
#singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.02-py3.SIF
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.10-py3.SIF
#pip install lime
pip install grad-cam seaborn
pip uninstall opencv-python
pip install opencv-python-headless
# show, grep, cut second word
gradcam_path="$(pip show grad-cam | grep "Location: " | cut -d " " -f 2)/pytorch_grad_cam/base_cam.py"

  # don't forget to edit pytorch_grad_cam/utils/image.py:L166 (don't resize)

#/auto/vestec1-elixir/home/petrmiculek/.local/lib/python3.8/site-packages/pytorch_grad_cam/utils/image.py
# or maybe better yet, edit:
# pytorch_grad_cam/base_cam.py:L-99999
# scaled = cam  # <orig_line>

# LIME generation
#CUDA_VISIBLE_DEVICE=0 python3 src/evaluate.py -r runs/colorful-breeze-45 --lime --limit 4 -w 1

# CAMs generation
# shellcheck disable=SC2157
if [ -z "wont-run" ]; then
  # sample eval run
  CUDA_VISIBLE_DEVICE=0 python3 src/evaluate.py -r runs/vivid-glitter-50 -w 0 -t 4 -b 1 --cam
  # full eval run
  CUDA_VISIBLE_DEVICE=0 python3 src/evaluate.py -r runs/vivid-glitter-50 -w 2 -b 8 --cam
  # extracting frames from videos
  singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.10-py3.SIF
  cp rose_videos.tar $SCRATCHDIR
  mkdir $SCRATCHDIR/vids
  tar -xf $SCRATCHDIR/rose_videos.tar --directory $SCRATCHDIR/vids --checkpoint=.1000
  mv $SCRATCHDIR/vids/RoseYoutu-full $SCRATCHDIR
  python scripts/extract_rose_youtu_videos.py -i $SCRATCHDIR/RoseYoutu-full -o $SCRATCHDIR/rose_youtu_imgs && cd $SCRATCHDIR && tar cf ryi6.tar --checkpoint=.1000 rose_youtu_imgs && cp ryi6.tar /storage/brno2/home/petrmiculek/facepad/

  # example output:
  # perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_test_all_attacks.csv >dataset_lists/test
  # mv dataset_lists/test dataset_lists/dataset_rose_youtu_test_all_attacks.csv
  # perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_train_all_attacks.csv >dataset_lists/train
  # mv dataset_lists/train dataset_lists/dataset_rose_youtu_train_all_attacks.csv
  # perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_val_all_attacks.csv >dataset_lists/val
  # mv dataset_lists/val dataset_lists/dataset_rose_youtu_val_all_attacks.csv
  #/mnt/storage-brno2/home/petrmiculek/RoseYoutu-full
  tar -xf ryi5.tar --directory ryi5 --checkpoint=.1000
  python3 src/train.py - p $SCRATCHDIR/dataset -e 15 -b 1 -w 2 -l 0.00001
fi
