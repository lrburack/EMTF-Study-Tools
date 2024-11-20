import os
import matplotlib.pyplot as plt
import config
import numpy as np
import pickle
import sys
from helpers import unique_name

# from Performance.efficiency import *
# from Performance.rate import *
from Performance.resolution import *
# from Performance.scale_factor import *


paths = np.array([
    os.path.join(config.STUDY_DIRECTORY, f"Tests/Condor/FullFlow/mode=15_prediction.pkl"),
    os.path.join(config.STUDY_DIRECTORY, "Tests/Condor/FullFlow/mode=14_prediction.pkl"),
    os.path.join(config.STUDY_DIRECTORY, "Tests/Condor/FullFlow/mode=13_prediction.pkl"),
    os.path.join(config.STUDY_DIRECTORY, "Tests/Condor/FullFlow/mode=11_prediction.pkl"),
])

labels = ["Mode 15", "Mode 14", "Mode 13", "Mode 11"]

fig_dir = "Tests/NewPerformanceCode/"
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

fig_name = f"allmodes"

predicted_pts = np.empty((len(paths)), dtype=object)
true_pts = np.empty((len(paths)), dtype=object)
eta = np.empty((len(paths)), dtype=object)
for i in range(len(paths)):
    with open(paths[i], "rb") as file:
        prediction_dict = pickle.load(file)
    with open(os.path.join(config.DATASET_DIRECTORY, prediction_dict["testing_dataset"], config.WRAPPER_DICT_NAME), "rb") as file:
        dataset = pickle.load(file)["dataset"]

    predicted_pts[i] = prediction_dict["predicted_pt"]
    true_pts[i] = dataset.get_features("gen_pt")[prediction_dict["testing_tracks"]]
    eta = dataset.get_features("gen_eta")[prediction_dict["testing_tracks"]]

fig, ax = resolution(predicted_pts, true_pts, labels)
plt.savefig(unique_name(f"resolution_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)