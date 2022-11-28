#!/usr/bin/env bash

for z in *.zip
do
  unzip -q "$z" &
done
