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
#   "Label"         : "Prediction/name"
    # "Current EMTF": f"BDT2025/Control/TrackPt/mode={mode}_prediction.pkl",
    # "New Training": f"BDT2025/Control/mode={mode}_prediction.pkl",
    "New Training": f"BDT2025/Control/NoGMTConversion/mode={mode}_prediction.pkl",
    "+ All Shower Info": f"BDT2025/ShowerStudy/AllShowerInfo/ChamberMatching/mode={mode}_prediction.pkl",
    # "- Outer FR": f"BDT2025/ShowerStudy/Realistic/RemoveBits/FR_outer_station/mode={mode}_prediction.pkl",
    # "- Outer FR + LooseShowerBit": f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly/mode={mode}_prediction.pkl",
    # "- Outer FR + LooseShowerBit + Promote 2Lor1N": f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/mode={mode}_prediction.pkl",
    "+ Loose Shower Count": f"BDT2025/ShowerStudy/LooseShowerCount/ChamberMatching/mode={mode}_prediction.pkl",
    # "+ Loose Shower Bit": f"BDT2025/ShowerStudy/LooseShowerBit/ChamberMatching/mode={mode}_prediction.pkl",
    # "1 Loose Promotion": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/1Loose/mode={mode}_prediction.pkl",
    # "2 Loose Promotion": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose/mode={mode}_prediction.pkl",
    # "2L or 1N Promotion": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal/mode={mode}_prediction.pkl",
    # "2L or 1N Promotion": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal/mode={mode}_prediction.pkl",
    # "2L or 1N Promotion + Loose Shower Bit": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal_and_LooseShowerBit/mode={mode}_prediction.pkl",
    # "2L or 1N Promotion + Loose Shower Count": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/2Loose_or_1Nominal_and_LooseShowerCount/mode={mode}_prediction.pkl",
    # "1 Nominal Promotion": f"BDT2025/ShowerStudy/Promotion/ChamberMatching/1Nominal/mode={mode}_prediction.pkl",
    # "Current EMTF": f"BDT2025/DebugTestingDistribution/traintest_with_testing_distribution/TrackPt/mode={mode}_prediction.pkl",
    # "New Training": f"BDT2025/DebugTestingDistribution/traintest_with_testing_distribution/mode={mode}_prediction.pkl",
}
# The output directory and figure name
# fig_dir = f"ReproducePreviousBDT/OldCode/Compressed/mode={mode}"
# fig_dir = f"ReproducePreviousBDT/BeatIt/BDTParams/esitmators_800_depth_5/mode={mode}"
# fig_dir = f"ReproducePreviousBDT/BeatIt/EventWeights/1overpT/mode={mode}"
fig_dir = f"Tests/MassFigureProduction/"
# fig_dir = f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/mode={mode}"
# fig_dir = f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly/mode={mode}"
# fig_dir = f"BDT2025/ShowerStudy/ChamberMatching/Promotion/2Loose_or_1Nominal_and_LooseShowerCount/mode={mode}"
# fig_dir = f"BDT2025/DebugTestingDistribution/traintest_with_testing_distribution/mode={mode}"
fig_name = f"mode={mode}"
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

    predicted_pts[i] = prediction_dict["predicted_pt"]
    if "gen_pt" in dataset.feature_names:
        true_pts[i] = dataset.get_features("gen_pt")[prediction_dict["testing_tracks"]]
        eta[i] = dataset.get_features("gen_eta")[prediction_dict["testing_tracks"]]
    


# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------


a = np.array([true_pts[0]] + list(predicted_pts), dtype=object)
fig, ax = pt_distribution(a, ["gen pt"] + labels)
fig.suptitle(fig_name)
plt.savefig(unique_name(f"pt_distribution_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# pt_cuts = [0, 10, 20, 30, 50, 100, 250, 500, 750, 1000]
# # pt_cuts = [0, 10, 20, 30, 50, 100, 250, 1000]
# eta_cuts = [1.2, 1.6, 2.1, 2.5]

# for j in range(len(eta_cuts) - 1):
#     for i in range(len(pt_cuts) - 1):
#         masked_predicted_pts = np.empty(len(predicted_pts), dtype=object)
#         masked_true_pts = np.empty(len(predicted_pts), dtype=object)
#         for k in range(len(predicted_pts)):
#             pt_mask = (true_pts[k] > pt_cuts[i]) & (true_pts[k] < pt_cuts[i + 1])
#             eta_mask = (np.abs(eta[k]) > eta_cuts[j]) & (np.abs(eta[k]) < eta_cuts[j + 1])
#             masked_predicted_pts[k] = predicted_pts[k][np.logical_and(pt_mask, eta_mask)]
#             masked_true_pts[k] = true_pts[k][np.logical_and(pt_mask, eta_mask)]

#         fig, ax = resolution(masked_predicted_pts, masked_true_pts, labels, resolution_bins=np.linspace(-1.5, 1.5, 151))
#         name = f"resolution_{pt_cuts[i]}<pT<{pt_cuts[i+1]}_{eta_cuts[j]}<eta<{eta_cuts[j + 1]}_{fig_name}"
#         fig.suptitle(name)
#         plt.savefig(unique_name(f"{name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

#     masked_predicted_pts = np.empty(len(predicted_pts), dtype=object)
#     masked_true_pts = np.empty(len(predicted_pts), dtype=object)
#     for k in range(len(predicted_pts)):
#         eta_mask = (np.abs(eta[k]) > eta_cuts[j]) & (np.abs(eta[k]) < eta_cuts[j + 1])
#         masked_predicted_pts[k] = predicted_pts[k][eta_mask]
#         masked_true_pts[k] = true_pts[k][eta_mask]
    
#     # Efficiency plot
#     fig, [low_pt, high_pt] = efficiency_split_low_high(masked_predicted_pts, masked_true_pts, labels, pt_cut=22)
#     fig.suptitle(fig_name)
#     plt.savefig(unique_name(f"efficiency_{eta_cuts[j]}<eta<{eta_cuts[j + 1]}_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
#     plt.close('all')

# cuts = {
#     r"$p_T$": [0, 10],
#     "eta": [1.2, 1.6],
# }

# cut_text = f"mode={mode}" + "\n"
# cut_text += "\n".join(
#     f"{cut_range[0]} < {var} < {cut_range[1]}" for var, cut_range in cuts.items()
# )

# bins = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
#            20,22,24,26,28,30,32,34,36,38,40,42,
#            44,46,48,50,60,70,80,90,100,150,200,
#            250,300,400,500,600,700,800,900,1000])
# # model_compare_all_pT_distribution([predicted_pts[0], predicted_pts[2]], [true_pts[0], true_pts[2]], [labels[0], labels[2]], pt_cut=22, pt_bins=bins)
# # plt.savefig(unique_name(f"model_comparison_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
# # model_compare_all_pT_distribution([predicted_pts[0], predicted_pts[2]], [true_pts[0], true_pts[2]], [labels[0], labels[2]], pt_cut=22, pt_bins=np.linspace(0, 50, 26))
# # plt.savefig(unique_name(f"model_comparison_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# # fig, ax = model_compare_control_pT_distribution(predicted_pts[0], predicted_pts[-1], true_pts[0], labels=[labels[0], labels[-1]], pt_cut=22)
# # plt.savefig(unique_name(f"model_comparison_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
# # fig, ax = model_compare_control_pT_distribution(predicted_pts[0], predicted_pts[-1], true_pts[0], labels=[labels[0], labels[-1]], pt_cut=22, pt_bins=bins)
# # plt.savefig(unique_name(f"model_comparison_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# fig, [low_pt, high_pt] = efficiency_split_low_high(predicted_pts, true_pts, labels, pt_cut=22)
# fig.suptitle(fig_name)
# annotate_cut_info(low_pt, cut_text, box_position=[0.05, 0.95])
# plt.savefig(unique_name(f"efficiency_split_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# fig, [ax1, ax2] = efficiency_with_ratio(predicted_pts, true_pts, labels=labels, pt_cut=22, pt_bins=np.linspace(0, 50, 26))
# ax1.set_ylim([0, 1])
# # ax2.set_ylim([.8, 1.2])
# fig.suptitle(fig_name)
# annotate_cut_info(ax1, cut_text, box_position=[0.96, 0.95])
# plt.savefig(unique_name(f"efficiency_with_ratio_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# fig, [ax1, ax2] = efficiency_with_ratio(predicted_pts, true_pts, labels=labels, pt_cut=22)
# ax1.set_ylim([0, 1])
# ax2.set_ylim([.8, 1.2])
# fig.suptitle(fig_name)
# annotate_cut_info(ax1, cut_text, box_position=[0.96, 0.95])
# plt.savefig(unique_name(f"efficiency_with_ratio_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# # fig, [ax1, ax2] = efficiency_with_difference(predicted_pts, true_pts, labels=labels, pt_cut=22, pt_bins=np.linspace(0, 50, 26))
# # ax1.set_ylim([0, 1])
# # fig.suptitle(fig_name)
# # plt.savefig(unique_name(f"efficiency_with_difference_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# fig, [ax1, ax2] = efficiency_with_difference(predicted_pts, true_pts, labels=labels, pt_cut=22)
# annotate_cut_info(ax1, cut_text, box_position=[0.95, 0.4])
# ax1.set_ylim([0, 1])
# fig.suptitle(fig_name)
# plt.savefig(unique_name(f"efficiency_with_difference_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)