#!/bin/bash

# Read the arguments passed to the script
CODE_DIRECTORY=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
UNIQUE_NAME=$4


echo "Running job to process: $UNIQUE_NAME"
# We will output to the scratch directory created by condor and then move the root file after
python3 $CODE_DIRECTORY/root_file_predict/root_file_predict.py $INPUT_DIR .

# Move the log to the scratch directory
mv $CODE_DIRECTORY/root_file_predict/logs/$UNIQUE_NAME/root_file_predict.log .