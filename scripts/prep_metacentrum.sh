# singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.02-py3.SIF

# copy files to scratch dir (memory-mapped)
cp -r s_brno2/data.tar "$SCRATCHDIR"
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
  #  export FIND='..'  # if dataset never overwritten
  first_lines="$(head dataset_lists/dataset_*.csv -n 3 | tail -n 1)"
  FIND="${first_lines%%client*}"
  export REPLACE="$SCRATCHDIR/data/"
fi

echo "# FIND=$FIND"
echo "# REPLACE=$REPLACE"

# edit copies of dataset annotation files
tmp_counter=1
for file in dataset_lists/dataset_*.csv; do
#	echo "# Processing paths in $file"
	out_tmp="dataset_lists/tmp_$tmp_counter.csv"
	echo "perl -pe 's|\Q$FIND\E|$REPLACE|g' $file>$out_tmp"
	echo "mv $out_tmp $file"
	tmp_counter=$((tmp_counter+1))
done

# unused
#for file in dataset_lists/dataset_*; do
#	echo "perl -pe 's|\Q$FIND\E|$SCRATCHDIR|g' $file">
#done

exit 0

# example output:
# perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_test_all_attacks.csv >dataset_lists/test
# mv dataset_lists/test dataset_lists/dataset_rose_youtu_test_all_attacks.csv
# perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_train_all_attacks.csv >dataset_lists/train
# mv dataset_lists/train dataset_lists/dataset_rose_youtu_train_all_attacks.csv
# perl -pe 's|\Q..\E|/dev/shm/scratch.shm/petrmiculek/job_14866782.meta-pbs.metacentrum.cz|g' dataset_lists/dataset_rose_youtu_val_all_attacks.csv >dataset_lists/val
# mv dataset_lists/val dataset_lists/dataset_rose_youtu_val_all_attacks.csv