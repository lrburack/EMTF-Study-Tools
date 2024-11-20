import argparse
import pickle
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True)
parser.add_argument("-c", "--code_directory", required=True)
args = parser.parse_args()

NAME = str(args.name)
CODE_DIRECTORY = str(args.code_directory)

# We need some of the methods from the parent directory. I suppose this works...
import sys
sys.path.insert(0, CODE_DIRECTORY)
from helpers import build_from_wrapper_dict, xrdcp
import config

dict_path = os.path.join(config.DATASET_DIRECTORY, NAME, config.WRAPPER_DICT_NAME)

# Fetch the wrapper dict, possibly from a remote location (eos)

print(f"* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t | Fetching wrapper dict to the condor sandbox...")
xrdcp(dict_path, config.WRAPPER_DICT_NAME)

print(f"* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t | Loading the wrapper dict to memory...")
with open(config.WRAPPER_DICT_NAME, "rb") as file:
    wrapper_dict = pickle.load(file)

# This is necessary because when pickle reloads the dataset object, each variable's references get messed up
wrapper_dict["dataset"].set_variable_references()

# Builds the dataset in-place
build_from_wrapper_dict(wrapper_dict)

print(f"* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t | Saving the wrapper dict to condor sandbox...")
# Save it to the condor sandbox
with open(config.WRAPPER_DICT_NAME, "wb") as file:
    pickle.dump(wrapper_dict, file)

# This code below is unnecessary because condor will move everything for us
# print(" * Writing the wrapper dict to the dataset directory...")
# # Send the wrapper dict back to where it came from, overwriting the unbuilt version
# xrdcp(config.WRAPPER_DICT_NAME, dict_path)

print(f"* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t | Done!")