import pickle
import os
import argparse
import config
from helpers import build_from_wrapper_dict, path_to_xrd
import shutil

from Dataset.Dataset import *
from Dataset.Default.Variables import *
from Dataset.Default.SharedInfo import *
from Dataset.Default.TrackSelectors import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--condor", required=False, default=0)
args = parser.parse_args()

CONDOR = bool(args.condor)

# --------------------------------------- CHANGE BELOW HERE -----------------------------------------

# base_dirs = config.wHMT
base_dirs = config.wHMT_testing_distribution
# base_dirs = config.EphemeralZeroBias

mode = 15

# name = "Control/Compressed/mode=" + str(mode)
name = "Control/Compressed/mode=" + str(mode) + "_testing_distribution"
# name = f"Control/Rates/Uncompressed/mode={mode}"

dataset = Dataset(variables=[
                            GeneratorVariables.for_mode(mode), 
                        #    RecoVariables.for_mode(mode), 
                            Theta.for_mode(mode),
                            St1_Ring2.for_mode(mode),
                            dPhi.for_mode(mode),
                            dTh.for_mode(mode),
                            FR.for_mode(mode),
                            RPC.for_mode(mode),
                            Bend.for_mode(mode),
                            # OutStPhi.for_mode(mode),
                            # dPhiSum4.for_mode(mode),
                            # dPhiSum4A.for_mode(mode),
                            # dPhiSum3.for_mode(mode),
                            # dPhiSum3A.for_mode(mode),
                            ],
                track_selector=TrackSelector(mode=mode, include_mode_15=True),
                shared_info=SharedInfo(mode=mode),
                compress=True
                )

# --------------------------------------- CHANGE ABOVE HERE -----------------------------------------

# Make sure you don't accidently overwrite an existing dataset
if os.path.exists(os.path.join(config.DATASET_DIRECTORY, name)) and os.path.isdir(os.path.join(config.DATASET_DIRECTORY, name)):
    print("A dateset with the name " + name + " has already been initiated.")
    overwrite = ""
    while overwrite not in ["y", "n"]:
        overwrite = input("Overwrite it (y/n)? ").lower()
    
    if overwrite == "n":
        exit()
    shutil.rmtree(os.path.join(config.DATASET_DIRECTORY, name))

wrapper_dict = {
    'dataset': dataset,
    'base_dirs': base_dirs,
    'files_per_endcap': 10
}

os.makedirs(os.path.join(config.DATASET_DIRECTORY, name), exist_ok=True)
dict_path = os.path.join(config.DATASET_DIRECTORY, name, config.WRAPPER_DICT_NAME)

if CONDOR:
    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)
    
    os.makedirs(os.path.join(config.CODE_DIRECTORY, "condor_wrapper/logs"), exist_ok=True)
    os.makedirs(os.path.join(config.CODE_DIRECTORY, "condor_wrapper/logs", name), exist_ok=True)
    condor_submit_path = os.path.join(config.CODE_DIRECTORY, "condor_wrapper/condor_wrapper.sub")
    command = f"condor_submit {condor_submit_path} code_directory={config.CODE_DIRECTORY} dataset_directory={config.DATASET_DIRECTORY} name={name} output_uri={path_to_xrd(os.path.join(config.DATASET_DIRECTORY, name))}"

    print(command)
    os.system(command)
else:
    # Builds the dataset in-place
    build_from_wrapper_dict(wrapper_dict)
    print(wrapper_dict["dataset"].data)

    print(dict_path)
    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)