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

# This file can be used only with EphemeralZeroBias data.
# For a given training, each mode must be prepared from the exact same set of events.

# ------------------------------------- EDIT BELOW THIS --------------------------------------
# Here is where you control what data will be used. 
# The labels on the left will be used in the figure legends

SQ = [15, 14, 13, 11]
DQ = [15, 14, 13, 11, 10, 9, 7]
OQ = [15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]

modes = SQ

# For each BDT training, there is a corresponding list of predictions for each mode.
predictions = {
#   "Label"         : "Prediction/name"
    "Current EMTF": [f"BDT2025/NewTraining/mode={mode}/TrackPt/Rates/mode={mode}_prediction.pkl" for mode in modes],
    "New Training": [f"BDT2025/NewTraining/mode={mode}/Rates/mode={mode}_prediction.pkl" for mode in modes],
}

# The output directory and figure name
# fig_dir = f"BDT2025/ShowerStudy/Realistic/ChamberMatching/"
fig_dir = f"BDT2025/ProposedImplementations/ChamberMatched_3StationLooseFRGainLooseShowerBit_Promote2LOR1N/scaled_SQ22"
fig_name = f"rate_SQ22"

pt_cut=22
# ------------------------------------- EDIT ABOVE THIS --------------------------------------

labels = list(predictions.keys())
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

# scale_factors = np.array(len(modes), dtype=object)
# for i in range(predictions):

# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------

for label in labels:
    rate = get_rate_zerobias_prediction([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], pt_cut=pt_cut, pt_scale_fxn=None)
    print(f"{label}: {rate} Hz")

# I have left this example if you want to use scale factors. 
# lm_pt = np.array([
#     get_leading_muon_pt([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], 
#                             scaling_constants=[scale_factors[mode] for mode in modes]),
#     get_leading_muon_pt([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], 
#                             scaling_constants=[[1.07, 0.015]] * len(modes)),
# ], dtype=object)

lm_pt = np.zeros(len(predictions), dtype=object)
for i in range(len(predictions)):
  lm_pt[i] = get_leading_muon_pt([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[labels[i]]])

fig, ax = rate_plot(lm_pt, labels=labels, pt_cut_to_annotate=pt_cut)
fig.suptitle(fig_name)
plt.savefig(unique_name(f"rate_plot_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

fig, [rate_ax, ratio_ax] = rate_plot_with_ratio(lm_pt, labels=labels, pt_cut_to_annotate=pt_cut)
fig.suptitle(fig_name)
# # Add the summary text box outside the plot area
plt.savefig(unique_name(f"rate_plot_with_ratio_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
