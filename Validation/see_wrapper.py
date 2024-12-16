import pickle
import sys
import numpy as np
import os
sys.path.insert(0, "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF_BDT")
import Dataset as Dataset
import config
from helpers import get_by_name

mode = int(sys.argv[1])

# dataset_name = "NewCode/Tests/AllZeros/fromcommandline/mode=15"
# dataset_name = f"ReproducePreviousBDT/Rates/mode={mode}"
# dataset_name = f"BDT2025/Control/mode={mode}_testing_distribution"
dataset_name = f"BDT2025/AllGenMuonsFromTestingDistribution/allgenmuons"
# dataset_name = "NewCode/Tests/AllZeros/mode=15"
with open(os.path.join(config.DATASET_DIRECTORY, dataset_name, config.WRAPPER_DICT_NAME), "rb") as file:
    data_builder = pickle.load(file)
print(data_builder.keys())
# print(data_builder["dataset"].feature_names)
# print(data_builder["files_per_endcap"])
print(data_builder["base_dirs"])
# print(data_builder["files_to_process"])
print(data_builder["dataset"].events_processed)
print(data_builder["dataset"].tracks_processed)
print(len(data_builder["dataset"].data))

# Compressed validation
# name1 = "Control/mode=" + str(mode) + "_testing_distribution"
# name2 = "ShowerDataset/ManyMode/mode=" + str(mode) + "_testing_distribution"

# name1 = "Control/mode=" + str(mode)
# name2 = "ShowerDataset/ManyMode/mode=" + str(mode)

# wrapper1 = get_by_name(name1)
# wrapper2 = get_by_name(name2)

# print(wrapper2["base_dirs"] == wrapper2["base_dirs"])
# print(wrapper1["base_dirs"])
# print(wrapper2["base_dirs"])
# # print(np.sum(new_dataset.get_features(["shower_type_0", "shower_type_1", "shower_type_2", "shower_type_3"]), axis=0))
# # print(np.sum(new_dataset.get_features(["careful_shower_bit_thresh=1", "careful_shower_bit_thresh=2", "careful_shower_bit_thresh=3"]), axis=0))

# # features = old_dataset.keys()

# # for feature in list(features):
#     print(feature + str())