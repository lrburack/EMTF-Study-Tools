import config
import pickle
import os
from Dataset import Dataset
import numpy as np
import time

def get_by_name(name : str, item : str = config.WRAPPER_DICT_NAME):
    path = os.path.join(config.RESULTS_DIRECTORY, name, item)
    with open(path, 'rb') as file:
        return pickle.load(file)

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
    print("Features:\t\t" + str(wrapper_dict['training_data_builder'].feature_names))
    print("Events to process:\t" + str(raw_data[list(raw_data.keys())[0]].GetEntries()))
    print("\n")
    print("------------------------------ Building Dataset ------------------------------")

    start_time = time.time()
    wrapper_dict['training_data_builder'].build_dataset(raw_data, wrapper_dict["tracks_to_process"])
    end_time = time.time()

    print("\n")
    print("------------------------------ Done Building! --------------------------------")
    print("Total time to build dataset: " + str(end_time-start_time))

    return wrapper_dict