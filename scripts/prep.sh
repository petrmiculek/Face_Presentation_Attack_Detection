#!/usr/bin/env bash

DS_DIR=~/dataset_lists

# create directories
# storing run logs
mkdir -p runs

# storing setup (e.g. default paths to datasets)
mkdir -p $DS_DIR


# playground - delete later
if 0; then
  exit 0

  # check that first argument is an existing file
  if [ ! -f "$1" ]; then
      echo "First argument must be an existing file"
      exit 1
  fi

  # activate pipenv



  cd test || exit
  SCRATCHDIR_ESCAPED=$(echo "$SCRATCHDIR" | sed 's/\//\\\//g')
  sed -i "s/../$SCRATCHDIR_ESCAPED/g" test_out.csv <test.csv

  # loop over files in $DS_DIR directory
  for file in "$DS_DIR"/*; do
      echo "$file"
  done

  # git delete local changes
  #  git checkout -- .
fi

