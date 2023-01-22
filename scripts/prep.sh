#!/usr/bin/env bash

# create directories
# storing run logs
mkdir -p runs

# storing setup (e.g. default paths to datasets)
mkdir -p config


# playground - delete later
if 0; then
  exit 0

  # check that first argument is an existing file
  if [ ! -f $1 ]; then
      echo "First argument must be an existing file"
      exit 1
  fi


  cd test
  SCRATCHDIR_ESCAPED=$(echo $SCRATCHDIR | sed 's/\//\\\//g')
  sed -i "s/../$SCRATCHDIR_ESCAPED/g" test_out.csv <test.csv

  # loop over files in config directory
  for file in config/*; do
      echo $file
  done

  # git delete local changes
  git checkout -- .
fi

