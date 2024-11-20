import config
import pickle
import os
from Dataset.Dataset import Dataset
import numpy as np
import time

def xrdcp(from_path, to_path):
    os.system(f"xrdcp -f {path_to_xrd(from_path)} {path_to_xrd(to_path)}")

def path_to_xrd(path):
    # If the path is eos, we need to reformat it for xrdcp file transfer
    if not path.startswith("/eos"):
        return path
    
    # The paths /eos/home-[LETTER]/[USER] and /eos/user/[LETTER]/[USER] lead to the same location, but we need it in the latter format to use xrd
    homestr = "/eos/home-"
    if path.startswith(homestr):
        letter = path[len(homestr)]
        user_path = path[len(homestr) + 2:]
        path = f"/eos/user/{letter}/{user_path}"
    
    # Return the path with the xrootd prefix
    return f"root://eosuser.cern.ch/{path}"

def get_by_name(name : str, item : str = config.WRAPPER_DICT_NAME):
    path = os.path.join(config.DATASET_DIRECTORY, name, item)
    with open(path, 'rb') as file:
        return pickle.load(file)

def unique_name(filename, directory="."):
    base_filepath = os.path.join(directory, filename + ".png")
    filepath = base_filepath
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{filename}_{counter}.png")
        counter += 1
    return filepath

def permute_together(*arrays):
    # Check that all arrays have the same length
    length = len(arrays[0])
    for array in arrays:
        if len(array) != length:
            raise ValueError("All arrays must have the same length.")
    
    # Generate a random permutation of indices
    permutation = np.random.permutation(length)
    
    # Apply the same permutation to each array
    for array in arrays:
        array[:] = array[permutation]

def build_from_wrapper_dict(wrapper_dict):
    raw_data, file_names = Dataset.get_root(base_dirs=wrapper_dict['base_dirs'], 
                                files_per_endcap=wrapper_dict['files_per_endcap'])

    print("------------------------------ Dataset Details -------------------------------")
    print("Features:\t\t" + str(wrapper_dict['dataset'].feature_names))
    print("Events to process:\t" + str(raw_data[list(raw_data.keys())[0]].GetEntries()))
    print("\n")
    print("------------------------------ Building Dataset ------------------------------")

    start_time = time.time()
    wrapper_dict['dataset'].build_dataset(raw_data)
    end_time = time.time()

    print("\n")
    print("------------------------------ Done Building! --------------------------------")
    print("Total time to build dataset: " + str(end_time-start_time))

    return wrapper_dict