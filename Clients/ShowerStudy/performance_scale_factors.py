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

mode = int(sys.argv[1])

# ------------------------------------- EDIT BELOW THIS --------------------------------------
# Here is where you control what data will be used. 
# The labels on the left will be used in the figure legends
predictions = {
#   "Label"         : "Prediction/name"
    # "Current EMTF": f"BDT2025/NewTraining/mode={mode}/TrackPt/mode={mode}_prediction.pkl",
    # "Current EMTF": f"BDT2025/NewTraining/mode={mode}/mode={mode}_prediction.pkl",
    "Current EMTF": f"BDT2025/ShowerStudy/Realistic/ChamberMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N/mode={mode}_prediction.pkl",
    # "New Training": f"BDT2025/NewTraining/ForcedModeTesting/mode={mode}/mode={mode}_prediction.pkl",
}
# The output directory and figure name
fig_dir = f"Tests/ScaleFactors/ForShowerStudy/again_mode={mode}"
fig_name = f"mode={mode}"

# if os.path.exists(os.path.join(config.FIGURE_DIRECTORY, fig_dir)):
#     shutil.rmtree(os.path.join(config.FIGURE_DIRECTORY, fig_dir))

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

# a = np.array([true_pts[0]] + list(predicted_pts), dtype=object)
# fig, ax = pt_distribution(a, [r"GEN $p_T$"] + labels)
# fig.suptitle(fig_name)
# plt.savefig(unique_name(f"pt_distribution_{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

pt_cut = 22

# pt_cuts = [0, 10, 20, 30, 50, 100, 250, 500, 750, 1000]
pt_cuts = [0, 10, 20, 30, 50, 100, 250, 1000]
eta_cuts = [1.2, 1.6, 2.1, 2.5]
bins = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000])

start = 5
stop = 500

pt_cuts_to_fit_sf = np.arange(5, 30)

# bins = np.linspace(start, stop, int((stop - start) / 3))
# scale_factor_fit, (pt_cuts_to_fit_sf, scale_factors), (pt_bins_to_fit_efficiency, efficiency_fit) = get_scale_factor_from_fit(true_pts[0], predicted_pts[0], pt_bins_to_fit_efficiency=bins, pt_cuts_to_fit_sf=pt_cuts_to_fit_sf)
# print(f"Scale factor fit: {scale_factor_fit}")
# print(f"Efficiency fit: {efficiency_fit}")

scale_factor_fit_simple, (pt_cuts_to_fit_sf_simple, scale_factors_simple) = get_scale_factor_simple(true_pts[0], predicted_pts[0], pt_cuts_to_fit_sf=pt_cuts_to_fit_sf, gen_pt_efficiency_bin_width=3)
print(f"Scale factor fit: {scale_factor_fit_simple}")

# fig, [low, high] = efficiency_split_low_high(predicted_pts, true_pts, labels, pt_cut=22)
# x = np.arange(1000)
# low.plot(x, theoretical_efficiency(x, *efficiency_fit), color="black")
# high.plot(x, theoretical_efficiency(x, *efficiency_fit), color="black")
# plt.savefig(unique_name(f"efficiency_fit_{fig_name}_{int(bins[0])}<pt_fit<{int(bins[-1])}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# fit_scaled_predicted_pts = np.empty(len(predicted_pts), dtype=object)
# simple_scaled_predicted_pts = np.empty(len(predicted_pts), dtype=object)
# for i in range(len(predicted_pts)):
#     fit_scaled_predicted_pts[i] = predicted_pts[i] * generic_scale_pt(predicted_pts[i], *scale_factor_fit)
#     simple_scaled_predicted_pts[i] = predicted_pts[i] * generic_scale_pt(predicted_pts[i], *scale_factor_fit_simple)

scaled_predicted_pts = np.array([
    # predicted_pts[0] * generic_scale_pt(predicted_pts[i], *scale_factor_fit),
    predicted_pts[0] * generic_scale_pt(predicted_pts[i], *scale_factor_fit_simple)
])

true_pts = np.array([
    # true_pts[0],
    true_pts[0]
])

pt_cuts = [5, 10, 15, 22]
for pt_cut in pt_cuts:
    fig, [low, high] = efficiency_split_low_high(scaled_predicted_pts, true_pts, ["with efficiency fit", "simple"], pt_cut=pt_cut)
    low.axhline(0.9, color="red", linestyle="dashed", zorder=-10)
    plt.savefig(unique_name(f"efficiency_scaled_pt_cut={pt_cut}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)


fig, ax = plt.subplots(1)
ax.plot(pt_cuts_to_fit_sf, current_EMTF_scale_pt(pt_cuts_to_fit_sf), label="current emtf")
# ax.plot(pt_cuts_to_fit_sf, scale_factors, label="with efficiency fit")
# ax.plot(pt_cuts_to_fit_sf, generic_scale_pt(pt_cuts_to_fit_sf, *scale_factor_fit), label="fit of with efficiency fit")
ax.scatter(pt_cuts_to_fit_sf, scale_factors_simple, label="simple")
ax.plot(pt_cuts_to_fit_sf, generic_scale_pt(pt_cuts_to_fit_sf, *scale_factor_fit_simple), label="fit of simple")
# ax.plot(pt_cuts_to_fit_sf, generic_scale_pt(pt_cuts_to_fit_sf, *scale_factor_fit), label="reproduction fit")
ax.set_xlabel(r"BDT $p_T$")
ax.set_ylabel("Scale factor")
ax.legend(loc='upper left')
# annotate_cut_info(ax, rf"fit a/(1-b*pT): \na={scale_factor_fit_simple[0]:.2f}\nb={scale_factor_fit_simple[1]:.5f}", box_position=(0.97, 0.05))
plt.savefig(unique_name(f"scale_factor{fig_name}", directory = os.path.join(config.FIGURE_DIRECTORY, fig_dir)), dpi=300)

# Save SF
sf_path = paths[0].split("_prediction.pkl")[0] + "binwidth=3_simple_sf.pkl"
with open(sf_path, "wb") as file:
    pickle.dump(scale_factor_fit_simple, file)

# sf_path = os.path.join(config.STUDY_DIRECTORY, list(predictions.keys())[0].split("_prediction.pkl")[0] + "efficiencyfit_sf.pkl")
# with open(sf_path, "wb") as file:
#     pickle.dump(scale_factor_fit, file)