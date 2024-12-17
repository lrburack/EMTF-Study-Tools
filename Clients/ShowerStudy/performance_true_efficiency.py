import os
import matplotlib.pyplot as plt
import config
import numpy as np
import pickle
import sys
from helpers import unique_name
import shutil

from Performance.efficiency import *  # The harmless segmentation fault comes from importing scipy packages here. Idk why
from Performance.model_insights import *
from Performance.rate import *
from Performance.resolution import *
from Performance.scale_factor import *
from Performance.pt_distribution import *

# ------------------------------------- EDIT BELOW THIS --------------------------------------
# We want to calculate efficiencies against all of the muons that EMTF is supposed to trigger on

all_gen_muon_dataset = "BDT2025/AllGenMuonsFromTestingDistribution/allgenmuons/exclude_pos79/wrapper_dict.pkl"

SQ = [15, 14, 13, 11]
DQ = [15, 14, 13, 11, 10, 9, 7]
OQ = [15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]

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

modes = OQ

predictions = {
#   "Label"         : "Prediction/name"
    "Current EMTF": [f"BDT2025/NewTraining/mode={mode}/TrackPt/mode={mode}_prediction.pkl" for mode in modes],
    # "New Training": [f"BDT2025/NewTraining/mode={mode}/mode={mode}_prediction.pkl" for mode in modes],
    "Proposed": [
        f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/mode={mode}_prediction.pkl" for mode in SQ
        ] + [
        f"BDT2025/NewTraining/mode={mode}/mode={mode}_prediction.pkl" for mode in modes if mode not in SQ
    ],
}
# Multiple of these tracks may be from the same event (this is encoded in event_correspondance).
# To deal with this should find the leading muon pt for each event, as is done in the get_rate method,
# and use the event_correspondance to figure out which of these corresponds to which gen muon (there is only one gen muon per event)

# The output directory and figure name
fig_dir = f"Tests/DidIBreakIt/"
# fig_dir = f"BDT2025/ProposedImplementations/ChamberMatched_3StationLooseFRGainLooseShowerBit_Promote2LOR1N/scaled_OQ3"
fig_name = "scaled_OQ3"

pt_cut = 3

# if os.path.exists(os.path.join(config.FIGURE_DIRECTORY, fig_dir)):
#     shutil.rmtree(os.path.join(config.FIGURE_DIRECTORY, fig_dir))

# ------------------------------------- EDIT ABOVE THIS --------------------------------------

print("Loading gen muons...")
with open(os.path.join(config.DATASET_DIRECTORY, all_gen_muon_dataset), "rb") as file:
    all_gen_muon_dict = pickle.load(file)
    all_gen_muon_dataset = all_gen_muon_dict["dataset"]

# print(all_gen_muon_dataset.get_features("gen_pt"))
true_pts = np.broadcast_to(all_gen_muon_dataset.get_features("gen_pt"), (len(predictions), all_gen_muon_dataset.events_processed))
eta = np.broadcast_to(all_gen_muon_dataset.get_features("gen_eta"), (len(predictions), all_gen_muon_dataset.events_processed))
print(f"Gen muon count: {all_gen_muon_dataset.events_processed}, {all_gen_muon_dataset.tracks_processed}")

labels = list(predictions.keys())
os.makedirs(os.path.join(config.FIGURE_DIRECTORY, fig_dir), exist_ok=True)

lm_pts = np.zeros((len(predictions), all_gen_muon_dataset.events_processed), dtype=object)

for j, key in enumerate(labels):
    paths = [os.path.join(config.STUDY_DIRECTORY, predictions[key][i]) for i in range(len(modes))]

    pt = np.empty(len(paths), dtype=object)
    event_correspondance = np.empty(len(paths), dtype=object)

    gen_pt_check = np.zeros(all_gen_muon_dataset.tracks_processed)

    print(f"Loading {key}...")
    for i in range(len(paths)):
        print(paths[i])
        with open(paths[i], "rb") as file:
            prediction_dict = pickle.load(file)
            # print(prediction_dict["files_per_endcap"])
        dataset = prediction_dict["testing_dataset"]

        if dataset.events_processed == 18900000:
            print("File discrepancy. Removing pos79")
            mask = (dataset.event_correspondance < 7300000) | (dataset.event_correspondance >= 7400000)
            
            pt[i] = prediction_dict["predicted_pt"][mask]
            event_correspondance[i] = np.concatenate(
                (dataset.event_correspondance[dataset.event_correspondance < 7300000], 
                dataset.event_correspondance[dataset.event_correspondance >= 7400000] - 100000)
            )
            gen_pt_check[event_correspondance[i]] = dataset.get_features("gen_pt")[mask]
        else:
            pt[i] = prediction_dict["predicted_pt"]
            event_correspondance[i] = dataset.event_correspondance
            gen_pt_check[event_correspondance[i]] = dataset.get_features("gen_pt")

        if i == 0:
            pt[i] = pt[i] * current_EMTF_scale_pt(pt[i])
        if i == 1:
            pt[i] = pt[i] * generic_scale_pt(pt[i], *scale_factors[modes[i]])
        # if dataset.events_processed != all_gen_muon_dataset.events_processed:
        #     raise ValueError(f"All datasets must have the same number of events. {dataset.events_processed}, {all_gen_muon_dataset.events_processed}")

    if len(np.where((gen_pt_check != true_pts[0]) & (gen_pt_check != 0))[0]) != 0:
        print(np.where((gen_pt_check != true_pts[0]) & (gen_pt_check != 0)))
        print("Warning: these datasets are not properly lined up")

    pt = np.concatenate(pt)
    event_correspondance = np.concatenate(event_correspondance)

    leading_muon_pt, events = lm_pt(pt, event_correspondance)
    print(f"{key}: {len(leading_muon_pt)} events with tracks of this quality")
    lm_pts[j][events] = leading_muon_pt


predicted_pts = lm_pts

# -------------------------------- CALL FUNCTIONS TO CREATE FIGURES BELOW HERE --------------------------------------

# eta_cuts = [1.2, 1.6, 2.1, 2.5]
bins = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000])

# Get eta masks
eta_cuts = [1.24, 2.4]
eta_masks = np.zeros((len(true_pts), len(eta_cuts) - 1), dtype=object)
for i in range(len(true_pts)):
    for j in range(len(eta_cuts) - 1):
        eta_masks[i][j] = (np.abs(eta[i]) >= eta_cuts[j]) & (np.abs(eta[i]) < eta_cuts[j + 1])

for i in range(len(eta_cuts) - 1):

    # With only eta cuts
    cut_text = f"{fig_name}\n{eta_cuts[i]} < eta < {eta_cuts[i+1]}"
    save_name_suffix = f"{eta_cuts[i]}<eta<{eta_cuts[i + 1]}_{fig_name}"
    masked_predicted_pts = np.empty(len(predicted_pts), dtype=object)
    masked_true_pts = np.empty(len(predicted_pts), dtype=object)
    for k in range(len(predicted_pts)):
        masked_predicted_pts[k] = predicted_pts[k][eta_masks[k][i]]
        masked_true_pts[k] = true_pts[k][eta_masks[k][i]]


    # fit_scaled_predicted_pts = np.empty(len(predicted_pts), dtype=object)
    # simple_scaled_predicted_pts = np.empty(len(predicted_pts), dtype=object)
    # for i in range(len(predicted_pts)):
        # fit_scaled_predicted_pts[i] = predicted_pts[i] * generic_scale_pt(predicted_pts[i], *scale_factor_fit)
        # simple_scaled_predicted_pts[i] = predicted_pts[i] * generic_scale_pt(predicted_pts[i], *scale_factor_fit_simple)
    # Split low high efficiency plot
    fig, [low_pt, high_pt] = efficiency_split_low_high(masked_predicted_pts, masked_true_pts, labels, pt_cut=pt_cut)
    low_pt.legend(loc='upper left')
    annotate_cut_info(low_pt, cut_text, box_position=[0.96, 0.1])
    plt.savefig(unique_name(f"efficiency_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    # efficiency with ratio plot
    fig, [ax1, ax2] = efficiency_with_ratio(masked_predicted_pts, masked_true_pts, labels=labels, pt_cut=pt_cut, pt_bins=np.linspace(0, 50, 26))
    ax1.set_ylim([0, 1])
    ax1.legend(loc='lower right')
    annotate_cut_info(ax1, cut_text, box_position=[0.04, 0.95])
    plt.savefig(unique_name(f"efficiency_with_ratio_lowpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
    fig, [ax1, ax2] = efficiency_with_ratio(masked_predicted_pts, masked_true_pts, labels=labels, pt_cut=pt_cut)
    ax1.set_ylim([0, 1])
    ax2.set_ylim([.8, 1.2])
    ax1.legend(loc='lower right')
    annotate_cut_info(ax1, cut_text, box_position=[0.97, 0.25])
    plt.savefig(unique_name(f"efficiency_with_ratio_highpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    # efficiency with difference plot
    fig, [ax1, ax2] = efficiency_with_difference(masked_predicted_pts, masked_true_pts, labels=labels, pt_cut=pt_cut, pt_bins=np.linspace(0, 50, 26))
    ax1.legend(loc='lower right')
    annotate_cut_info(ax1, cut_text, box_position=[0.04, 0.95])
    ax1.set_ylim([0, 1])
    plt.savefig(unique_name(f"efficiency_with_difference_lowpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
    fig, [ax1, ax2] = efficiency_with_difference(masked_predicted_pts, masked_true_pts, labels=labels, pt_cut=pt_cut)
    ax1.legend(loc='lower right')
    annotate_cut_info(ax1, cut_text, box_position=[0.97, 0.25])
    ax1.set_ylim([0, 1])
    plt.savefig(unique_name(f"efficiency_with_difference_highpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    plt.close('all')


    # Check to make sure we're fitting well
    # start = 5
    # stop = 1000
    # bins = np.linspace(start, stop, int((stop - start) / 3))
    # pt_cut = 22
    # efficiency, efficiency_err = get_efficiency(masked_true_pts[0], masked_predicted_pts[0], bins, pt_cut=pt_cut)
    # efficiency_fit = fit_efficiency(bins, efficiency, efficiency_err)
    # fig, [low, high] = efficiency_split_low_high(masked_predicted_pts, masked_true_pts, labels, pt_cut=pt_cut)
    # x = np.arange(1000)
    # low.plot(x, theoretical_efficiency(x, *efficiency_fit), color="black")
    # high.plot(x, theoretical_efficiency(x, *efficiency_fit), color="black")
    # plt.savefig(unique_name(f"efficiency_fit_{fig_name}_ptcut={pt_cut}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    # # CALCULATE SF
    # scale_factor_fit, scale_factor_info, efficiency_fit = get_scale_factor_from_fit(masked_true_pts[0], masked_predicted_pts[0], pt_bins_to_fit_efficiency=bins)
    # print(f"Scale factor fit: {scale_factor_fit}")
    # print(f"Efficiency fit: {efficiency_fit}")

    # scaled_predictions = np.array([
    #     masked_predicted_pts[0] * generic_scale_pt(masked_predicted_pts[0], *scale_factor_fit),
    # ])

    # # Test scaling
    # pt_cuts = [5, 10, 15, 22]
    # for pt_cut in pt_cuts:
    #     fig, [low, high] = efficiency_split_low_high(scaled_predictions, masked_true_pts, labels, pt_cut=pt_cut)
    #     low.axhline(0.9, color="red", linestyle="dashed", zorder=-10)
    #     plt.savefig(unique_name(f"efficiency_scaled_pt_cut={pt_cut}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

    # fig, ax = plt.subplots(1)
    # ax.plot(scale_factor_info[0], current_EMTF_scale_pt(scale_factor_info[0]), label="current scaling")
    # ax.plot(scale_factor_info[0], generic_scale_pt(scale_factor_info[0], *scale_factor_fit), label="new scaling")
    # ax.set_xlabel(r"BDT $p_T$")
    # ax.set_ylabel("Scale factor")
    # ax.legend()
    # plt.savefig(unique_name(f"scale_factor{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)


# cut_text = f"{fig_name}"

# save_name_suffix = f"{fig_name}"

# # Split low high efficiency plot
# fig, [low_pt, high_pt] = efficiency_split_low_high(predicted_pts, true_pts, labels, pt_cut=pt_cut)
# low_pt.legend(loc='upper left')
# annotate_cut_info(low_pt, cut_text, box_position=[0.96, 0.1])
# plt.savefig(unique_name(f"efficiency_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# # efficiency with ratio plot
# fig, [ax1, ax2] = efficiency_with_ratio(predicted_pts, true_pts, labels=labels, pt_cut=pt_cut, pt_bins=np.linspace(0, 50, 26))
# ax1.set_ylim([0, 1])
# ax1.legend(loc='lower right')
# annotate_cut_info(ax1, cut_text, box_position=[0.04, 0.95])
# plt.savefig(unique_name(f"efficiency_with_ratio_lowpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
# fig, [ax1, ax2] = efficiency_with_ratio(predicted_pts, true_pts, labels=labels, pt_cut=pt_cut)
# ax1.set_ylim([0, 1])
# ax2.set_ylim([.8, 1.2])
# ax1.legend(loc='lower right')
# annotate_cut_info(ax1, cut_text, box_position=[0.97, 0.25])
# plt.savefig(unique_name(f"efficiency_with_ratio_highpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# # efficiency with difference plot
# fig, [ax1, ax2] = efficiency_with_difference(predicted_pts, true_pts, labels=labels, pt_cut=pt_cut, pt_bins=np.linspace(0, 50, 26))
# ax1.legend(loc='lower right')
# annotate_cut_info(ax1, cut_text, box_position=[0.04, 0.95])
# ax1.set_ylim([0, 1])
# plt.savefig(unique_name(f"efficiency_with_difference_lowpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)
# fig, [ax1, ax2] = efficiency_with_difference(predicted_pts, true_pts, labels=labels, pt_cut=pt_cut)
# ax1.legend(loc='lower right')
# annotate_cut_info(ax1, cut_text, box_position=[0.97, 0.25])
# ax1.set_ylim([0, 1])
# plt.savefig(unique_name(f"efficiency_with_difference_highpT_{save_name_suffix}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

