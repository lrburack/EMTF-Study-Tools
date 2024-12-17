import pickle
import sys
import numpy as np
sys.path.insert(0, "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF_BDT")
import Dataset as Dataset
from helpers import get_by_name

old_dataset_name = "ShowerStudy/all_variables"
new_dataset_name = "ShowerStudy/validate_new_shower_code_again"

old_dataset = get_by_name(old_dataset_name)['training_data_builder']
new_dataset = get_by_name(new_dataset_name)['training_data_builder']

print(len(old_dataset.data))
print(len(new_dataset.data))

print(np.sum(old_dataset.filtered))
print(np.sum(new_dataset.filtered))

print(np.sum(np.logical_not(old_dataset.filtered == new_dataset.filtered)))

print(np.sum(new_dataset.get_features(["loose_3"]) != 0))

for feature in list(old_dataset.feature_names):
    rounded_old = old_dataset.get_features([feature])
    rounded_new = new_dataset.get_features([feature])
    mismatches = np.sum(np.logical_not(rounded_old == rounded_new))
    print(feature + ": " + str(mismatches))
    if mismatches != 0:
        print("\t" + str(rounded_old))
        print("\t" + str(rounded_new))
