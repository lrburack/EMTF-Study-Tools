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

SQ = [15, 14, 13, 11]
DQ = [15, 14, 13, 11, 10, 9, 7]
OQ = [15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]

modes = SQ

scale_factors = {
    15: [1.30468498, 0.00488588],
    14: [1.34241812, 0.00471303],
    13: [1.33422215, 0.00540374],
    12: [1.7994723, 0.0161077],
    11: [1.31059254, 0.00631599],
    10: [1.74349138, 0.0167764],
    9: [1.76879585, 0.0154091],
    7: [1.65059324, 0.02127335],
    6: [1.65991508, 0.02237423],
    5: [2.32058488, 0.02325334],
    3: [1, 0],
}

predictions = {
#   "Label"         : "Prediction/name"
    "Current EMTF": [f"BDT2025/NewTraining/mode={mode}/TrackPt/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "New Training": [f"BDT2025/NewTraining/mode={mode}/Rates/mode={mode}_prediction.pkl" for mode in modes],
    "Proposed": [
        f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/Rates/mode={mode}_prediction.pkl" for mode in SQ
        ] + [
        f"BDT2025/NewTraining/mode={mode}/Rates/mode={mode}_prediction.pkl" for mode in modes if mode not in SQ
    ],
    # "+ All Shower Info": [f"BDT2025/ShowerStudy/AllShowerInfo/ChamberMatching/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "- Outer FR": [f"BDT2025/ShowerStudy/Realistic/RemoveBits/FR_outer_station/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "- Outer FR + LooseShowerBit": [f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "- Outer FR + LooseShowerBit + Promote 2Lor1N": [f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "+ Loose Shower Count": [f"BDT2025/ShowerStudy/LooseShowerCount/ChamberMatching/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "+ Loose Shower Bit": [f"BDT2025/ShowerStudy/LooseShowerBit/ChamberMatching/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "1 Loose Promotion": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/1Loose/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "2 Loose Promotion": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "1 Nominal Promotion": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/1Nominal/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "2L or 1N Promotion": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "2L or 1N Promotion + LooseShowerCount": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal_and_LooseShowerCount/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "2L or 1N Promotion + LooseShowerBit": [f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal_and_LooseShowerBit/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "+ All Shower Info": [f"BDT2025/ShowerStudy/AllShowerInfo/ChamberMatching/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "Train <100GeV only": [f"ReproducePreviousBDT/BeatIt/EventWeights/lessthan100/Rates/mode={mode}_prediction.pkl" for mode in modes],
    # "Old Code": [f"ReproducePreviousBDT/OldCode/Rates/mode={mode}_prediction.pkl" for mode in modes],
}

# The output directory and figure name
# fig_dir = f"BDT2025/ShowerStudy/Realistic/ChamberMatching/"
fig_dir = f"BDT2025/ProposedImplementations/ChamberMatched_3StationLooseFRGainLooseShowerBit_Promote2LOR1N/scaled_SQ22"
fig_name = f"rate_scaled_SQ22"

pt_cut=22
# fig_name = f"mode={modes[0]}"
# ------------------------------------- EDIT ABOVE THIS --------------------------------------

labels = list(predictions.keys())
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

# scale_factors = np.array(len(modes), dtype=object)
# for i in range(predictions):

# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------

for label in labels:
    rate = get_rate_zerobias_prediction([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], pt_cut=pt_cut, pt_scale_fxn=None)
    print(f"{label}: {rate} Hz")

lm_pt = np.array([
    get_leading_muon_pt([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], 
                            scaling_constants=[scale_factors[mode] for mode in modes]),
    get_leading_muon_pt([os.path.join(config.STUDY_DIRECTORY, prediction) for prediction in predictions[label]], 
                            scaling_constants=[[1.07, 0.015]] * len(modes)),
], dtype=object)

print(lm_pt)

fig, ax = rate_plot(lm_pt, labels=labels, pt_cut_to_annotate=pt_cut)
fig.suptitle(fig_name)
plt.savefig(unique_name(f"rate_plot_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)


fig, [rate_ax, ratio_ax] = rate_plot_with_ratio(lm_pt, labels=labels, pt_cut_to_annotate=pt_cut)
fig.suptitle(fig_name)
# # Add the summary text box outside the plot area
plt.savefig(unique_name(f"rate_plot_with_ratio_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)