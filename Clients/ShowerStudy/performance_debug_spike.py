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
from Performance.pt_distribution import *

mode = int(sys.argv[1])

# ------------------------------------- EDIT BELOW THIS --------------------------------------
# Here is where you control what data will be used. 
# The labels on the left will be used in the figure legends
predictions = {
    # "Current EMTF": f"DebugSpike/TrackPt/mode={mode}_prediction.pkl",
    # "TrainWithEndcap": f"DebugSpike/TrainWithEndcap/mode={mode}_prediction.pkl",
    # "Shuffle Training": f"DebugSpike/ShuffleTraining/mode={mode}_prediction.pkl",
    "New Training": f"BDT2025/Control/mode={mode}_prediction.pkl",
}
# The output directory and figure name
# fig_dir = f"ReproducePreviousBDT/OldCode/Compressed/mode={mode}"
# fig_dir = f"ReproducePreviousBDT/BeatIt/BDTParams/esitmators_800_depth_5/mode={mode}"
# fig_dir = f"ReproducePreviousBDT/BeatIt/EventWeights/1overpT/mode={mode}"
# fig_dir = f"DebugSpike/FR/NewTraining/mode={mode}"
fig_dir = f"DebugTestingDistribution/Charge/mode={mode}"
fig_name = f"mode={mode}"
# ------------------------------------- EDIT ABOVE THIS --------------------------------------

labels = list(predictions.keys())
paths = [os.path.join(config.STUDY_DIRECTORY, predictions[key]) for key in labels]
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

predicted_pts = np.empty((len(paths)), dtype=object)
true_pts = np.empty((len(paths)), dtype=object)
eta = np.empty((len(paths)), dtype=object)

discriminator = np.empty((len(paths)), dtype=object)


for i in range(len(paths)):
    with open(paths[i], "rb") as file:
        prediction_dict = pickle.load(file)
    
    dataset = prediction_dict["testing_dataset"]

    # Set discriminator
    # discriminator[i] = np.any(dataset.get_features(["RPC_1", "RPC_2", "RPC_3", "RPC_4"]) == 1, axis=1)
    # discriminator[i] = dataset.get_features(["FR_1"]) == 1
    # discriminator[i] = dataset.get_features(["emtfTrack_endcap"]) == 1
    discriminator[i] = (dataset.get_features(["gen_q"]) == 1)[prediction_dict["testing_tracks"]]

    predicted_pts[i] = prediction_dict["predicted_pt"]
    if "gen_pt" in dataset.feature_names:
        true_pts[i] = dataset.get_features("gen_pt")[prediction_dict["testing_tracks"]]
        eta[i] = dataset.get_features("gen_eta")[prediction_dict["testing_tracks"]]
    

# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------

# print(get_rate_zerobias_prediction(paths, pt_cut=22))


# y_lims_mode = {15: [.85, 1], 14: [.7, 1], 13: [.7, 1], 11: [.7, 1]}
# y_lims = y_lims_mode[mode]
y_lims = [0.6, 1]

# a = np.array([true_pts[0]] + list(predicted_pts), dtype=object)

# fig, ax = pt_distribution(a, ["gen pt"] + labels)
# fig.suptitle(fig_name)
# plt.savefig(unique_name(f"pt_distribution_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

pt_cuts = [0, 10, 20, 30, 50, 100, 250, 1000]
eta_cuts = [1.2, 1.6, 2.1, 2.4]

labels = [f"Pos charge | {labels[0]}", f"Neg charge | {labels[0]}"]

for j in range(len(eta_cuts) - 1):
    for i in range(len(pt_cuts) - 1):
        masked_predicted_pts = np.empty(2, dtype=object)
        masked_true_pts = np.empty(2, dtype=object)

        pt_mask = (true_pts[0] > pt_cuts[i]) & (true_pts[0] < pt_cuts[i + 1])
        eta_mask = (np.abs(eta[0]) > eta_cuts[j]) & (np.abs(eta[0]) < eta_cuts[j + 1])

        print(discriminator, pt_mask, eta_mask)
        masked_predicted_pts[0] = predicted_pts[0][discriminator[0] & pt_mask & eta_mask]
        masked_true_pts[0] = true_pts[0][discriminator[0] & pt_mask & eta_mask]
        masked_predicted_pts[1] = predicted_pts[0][~discriminator[0] & pt_mask & eta_mask]
        masked_true_pts[1] = true_pts[0][~discriminator[0] & pt_mask & eta_mask]

        fig, ax = resolution(masked_predicted_pts, masked_true_pts, labels, resolution_bins=np.linspace(-2, 2, 201))
        name = f"resolution_{pt_cuts[i]}<pT<{pt_cuts[i+1]}_{eta_cuts[j]}<eta<{eta_cuts[j + 1]}_{fig_name}"
        fig.suptitle(name)
        plt.savefig(unique_name(f"{name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    # masked_predicted_pts = np.empty(len(predicted_pts), dtype=object)
    # masked_true_pts = np.empty(len(predicted_pts), dtype=object)
    # for k in range(len(predicted_pts)):
    #     eta_mask = (eta[k] > eta_cuts[j]) & (eta[k] < eta_cuts[j + 1])
    #     masked_predicted_pts[k] = predicted_pts[k][eta_mask]
    #     masked_true_pts[k] = true_pts[k][eta_mask]
    # # Efficiency plot
    # fig, [low_pt, high_pt] = split_low_high(masked_predicted_pts, masked_true_pts, labels, pt_cut=22)
    # fig.suptitle(fig_name)
    # high_pt.set_ylim(y_lims)
    # plt.savefig(unique_name(f"efficiency_{eta_cuts[j]}<eta<{eta_cuts[j + 1]}_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# discriminated_predicted_pt = np.empty((len(paths)), dtype=object)
# discriminated_true_pt = np.empty((len(paths)), dtype=object)
# for i in range(len(paths)):
#     discriminated_predicted_pt[i] = predicted_pts[i][discriminator[i]]
#     discriminated_true_pt[i] = true_pts[i][discriminator[i]]
# make_plots(discriminated_predicted_pt, discriminated_true_pt)

# discriminated_predicted_pt = np.empty((len(paths)), dtype=object)
# discriminated_true_pt = np.empty((len(paths)), dtype=object)
# for i in range(len(paths)):
#     discriminated_predicted_pt[i] = predicted_pts[i][discriminator[i]]
#     discriminated_true_pt[i] = true_pts[i][discriminator[i]]
# make_plots(discriminated_predicted_pt, discriminated_true_pt)
