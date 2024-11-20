#!/bin/bash

# Read the arguments passed to the script
CODE_DIRECTORY=$1
DATASET_DIRECTORY=$2
NAME=$3


echo "Running job to process: $NAME"
python3 $CODE_DIRECTORY/condor_wrapper/condor_wrapper.py -c $CODE_DIRECTORY -n $NAME

# Move the log to the scratch directory
mv $CODE_DIRECTORY/condor_wrapper/logs/$NAME/build_dataset.log .