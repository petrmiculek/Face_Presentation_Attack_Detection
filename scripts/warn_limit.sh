#!/bin/bash
# go through all .py files line by line, and warn if a line contains "limit = \d+"

# get all .py files
files=`find ../src -name "*.py"`

re="limit = [0-9]+"
retval=0
# go through all files
for file in $files
do
    # go through all lines
    while read line
    do
        # check if line contains "limit = \d+"
        if [[ $line =~ $re ]]
        then
            echo "WARNING: $file contains a limit: $line"
            retval=1
        fi
    done < "$file"
done

exit $retval
