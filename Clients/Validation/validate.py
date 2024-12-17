import pickle
import sys
import numpy as np
sys.path.insert(0, "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools")
import Dataset.Dataset as Dataset

mode = int(sys.argv[1])

# Uncompressed validation
# old_dataset_path = "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/Validation/training_data_m=" + str(mode) + ".pkl"
# new_dataset_path = "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/Control/mode=" + str(mode) + "/wrapper_dict.pkl"

# Compressed validation
# old_dataset_path = "/eos/home-l/lburack/work/BDT_studies/OldResults/Datasets/Validation/training_data_compressed_m=" + str(mode) + ".pkl"
# new_dataset_path = "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/ReproducePreviousBDT/ValidateNewCode/mode=" + str(mode) + "/wrapper_dict.pkl"
old_dataset_path = f"/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools/Validation/Results/AllModes/mode={mode}/training_data_m=" + str(mode) + ".pkl"
new_dataset_path = "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/Validate/AllModes/mode=" + str(mode) + "/wrapper_dict.pkl"

with open(old_dataset_path, 'rb') as file:
    old_dataset = pickle.load(file)

with open(new_dataset_path, 'rb') as file:
    new_dataset = pickle.load(file)['dataset']


print(list(old_dataset.keys()))
print(new_dataset.feature_names)

features = old_dataset.keys()

check_until = 100000000

for feature in list(features):
    rounded_old = np.array(old_dataset[feature]).flatten()[:check_until]
    rounded_new = new_dataset.get_features([feature])[:check_until]
    mismatches = np.sum(np.logical_not(rounded_old == rounded_new))

    # print(f"First mismatch: {np.where(rounded_old != rounded_new)[0][0]}")

    print(feature + ": " + str(mismatches))
    if mismatches != 0:
        print("\t" + str(rounded_old))
        print("\t" + str(rounded_new))