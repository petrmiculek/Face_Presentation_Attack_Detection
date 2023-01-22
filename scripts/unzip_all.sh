#!/usr/bin/env bash

# unzip all zip files in the current directory
for z in *.zip
do
  unzip -q "$z" &
done



