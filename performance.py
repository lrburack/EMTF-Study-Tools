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
predictions = {
#   "Label"         : "Prediction/name"
    "Control"      : f"Control/mode=15_prediction.pkl",
    "New Bend"      : f"NewBend/mode=15_prediction.pkl",
}
# The output directory and figure name
fig_dir = "NewBend/"
fig_name = f"mode=15"
# ------------------------------------- EDIT ABOVE THIS --------------------------------------

labels = list(predictions.keys())
paths = [os.path.join(config.STUDY_DIRECTORY, predictions[key]) for key in labels]
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

predicted_pts = np.empty((len(paths)), dtype=object)
true_pts = np.empty((len(paths)), dtype=object)
eta = np.empty((len(paths)), dtype=object)
for i in range(len(paths)):
    with open(paths[i], "rb") as file:
        prediction_dict = pickle.load(file)
    dataset = prediction_dict["testing_dataset"]

    # predicted_pts[i] = current_EMTF_scale_pt(prediction_dict["predicted_pt"]) * prediction_dict["predicted_pt"]
    predicted_pts[i] = prediction_dict["predicted_pt"]
    if "gen_pt" in dataset.feature_names:
        true_pts[i] = dataset.get_features("gen_pt")[prediction_dict["testing_tracks"]]
        eta[i] = dataset.get_features("gen_eta")[prediction_dict["testing_tracks"]]


# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------

# Efficiency plot
fig, [low_pt, high_pt] = split_low_high(predicted_pts, true_pts, labels, pt_cut=22)
plt.savefig(unique_name(f"efficiency_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)