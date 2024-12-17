import os
import matplotlib.pyplot as plt
import config
import numpy as np
import pickle
import sys
from helpers import unique_name

from Performance.efficiency import *  # The harmless segmentation fault comes from importing scipy packages here. Idk why
from Performance.model_insights import *
from Performance.rate import *
from Performance.resolution import *
from Performance.scale_factor import *

# ------------------------------------- EDIT BELOW THIS --------------------------------------
# Here is where you control what data will be used. 
# The labels on the left will be used in the figure legends
base_dataset_name = "Control/Rates/True"
predictions = {
#   "Label"         : "Prediction/name"
    "Mode 15": f"{base_dataset_name}/mode=15/wrapper_dict.pkl",
    "Mode 14": f"{base_dataset_name}/mode=14/wrapper_dict.pkl",
    "Mode 13": f"{base_dataset_name}/mode=13/wrapper_dict.pkl",
    "Mode 11": f"{base_dataset_name}/mode=11/wrapper_dict.pkl",
    # "Mode 15": "Control/Rates/Compressed/mode=15_prediction.pkl",
    # "Mode 14": "Control/Rates/Compressed/mode=14_prediction.pkl",
    # "Mode 13": "Control/Rates/Compressed/mode=13_prediction.pkl",
    # "Mode 11": "Control/Rates/Compressed/mode=11_prediction.pkl",
}
# The output directory and figure name
# fig_dir = "ShowerStudy/Uncompressed/LooseShowerBit"
# ------------------------------------- EDIT ABOVE THIS --------------------------------------

labels = list(predictions.keys())
paths = [os.path.join(config.DATASET_DIRECTORY, predictions[key]) for key in labels]

# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------


print(get_rate_zerobias_NTuples(paths, labels=labels, pt_cut=22))
# print(get_rate_zerobias_prediction(paths, pt_cut=22))