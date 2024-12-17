import pickle
import sys
import numpy as np
import os
sys.path.insert(0, "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools")
import config
import Dataset as Dataset
from helpers import get_by_name

mode = 14

path1 = os.path.join(config.STUDY_DIRECTORY, f"Control/AllStats/Uncompressed/mode={mode}_prediction.pkl")
path2 = os.path.join(config.STUDY_DIRECTORY, f"ReproducePreviousBDT/wHMT/mode={mode}_prediction.pkl")

with open(path1, 'rb') as file:
    prediction_dict1 = pickle.load(file)

print(prediction_dict1["testing_dataset"].tracks_processed)
# print(dataset.)

with open(path2, 'rb') as file:
    prediction_dict2 = pickle.load(file)

print(prediction_dict2)
with open(prediction_dict2["model_path"], 'rb') as file:
    model2 = pickle.load(file)

print(model2.target_to_pT)

print(len(prediction_dict1["testing_tracks"]))
print(len(prediction_dict2["testing_tracks"]))

print(np.all(prediction_dict1["testing_dataset"].data == prediction_dict2["testing_dataset"].data))