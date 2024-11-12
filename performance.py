import os
import matplotlib.pyplot as plt
import config
import numpy as np
import pickle
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import sys

from matplotlib.legend_handler import HandlerPathCollection
from helpers import get_by_name

mode = int(sys.argv[1])
print("MODE: " + str(mode))
# mode = 13

paths = np.array([
    "/eos/home-l/lburack/work/BDT_studies/Results/Studies/Control/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/ForPresentation_10_17_2024/ManyModeAllVariables/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/SectorMatching/ControlCheck/mode=" + str(mode) + "_prediction.pkl",
    "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/SectorMatching/AllVariables/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/SectorMatching/LooseShowerCount/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/ForPresentation_10_17_2024/ManyModeLooseShowerBit/mode=" + str(mode) + "_prediction.pkl",
    "/eos/home-l/lburack/work/BDT_studies/Results/Studies/ShowerStudy/SectorMatching/Promotion/2Loose_or_1Nominal_LooseShowerBit/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/Tests/RateCalculations/OneNtuple_v2/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/ControlCheck/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Redone/ControlCheck/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Redone/slope_1/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Redone/CheckUnset/slope_1/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Redone/pattern_1/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Redone/wire_1/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/ModeFeature/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/FirstTests/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/slope_pattern_1/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Top7Features/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Top5Features/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/Top3Features/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/CalculatedVarsExcluded/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/FullNtupleStudy/ControlCheck/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/NEstimators/800/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/NEstimators/1500/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/MaxDepth/15/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/MaxDepth/20/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/LossFunction/Linear/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/LossFunction/Linear/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/LossFunction/SquaredError/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log2_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log2+1_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log1.5_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log1.7_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log2.2_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/log2.5_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/EventWeight/uniform_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/Control/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/TrainingSubset/Eta/only_<1.6/mode=" + str(mode) + "_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/TrainingSubset/Eta/uniform_prediction.pkl",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Studies/MLStudy/TrainingSubset/Eta/uniform_prediction.pkl",
])

fig_dir = "ShowerStudy/SectorMatching/2Loose_or_1Nominal_LooseShowerBit"
# fig_dir = "MLStudy/EventWeights"
fig_name = "mode=" + str(mode)
# fig_name = "uniform_weights"

# labels = ["Control", "NN Baseline", "100 per layer", "3 layer"]
# labels = ["200 estimator", "3 depth", "500 bin", "Control"]
# labels = ["800 estimator", "10 depth", "2000 bin", "Control"]
# labels = ["Mode 15", "Mode 14", "Mode 13", "Mode 11", "Mode 10", "Mode 9", "Mode 7"]
# labels = ["Current EMTF", "All Features", "+ wire_2"]
# labels = ["Control", "Control Check"]
labels = ["Control", "All Variables", "2L || 1N + Loose Shower Bit"]
# labels = ["Current EMTF", "Uniform Weight"]
# labels = ["Current EMTF", "All Shower Info", "Loose Shower Bit", "Promote 1 Loose", "Promote 1 Nominal", "Promote 2 Loose", "Promote 2 Loose or 1 Nominal"]
# labels = ["Current EMTF", "All Shower Info", "Loose Shower Bit", "Promote 1 Loose", "Promote 1 Nominal", "Promote 2 Loose", "Promote 2 Loose or 1 Nominal"]
# labels = ["Current EMTF", "All Shower Info", "2L || 1N Promotion"]
# labels = ["Current EMTF", "BinaryBDT"]
# labels = ["Current EMTF"]

bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000]
pt_cut = 22

eta_bounds = None
# eta_bounds = [0, 1.6]
# eta_bounds = [1.6, 1000]

if eta_bounds != None and eta_bounds == [0, 1.6]:
    fig_name += "_eta<1.6"
if eta_bounds != None and eta_bounds[0] == 1.6:
    fig_name += "_eta>1.6"

# mode_weights = {
#     15: 0.38, 14: 0.12, 13: 0.02, 11: 0.05, 7: 0.08
# }
ylims_mode = {
    15: [.85, 1], 14: [.7, 1], 13: [.7, 1], 11: [.7, 1], 7: [0, .5], 12: [0, 1]
}
y_lims = ylims_mode[mode]
# y_lims = [.85, 1]
# y_lims = [0, 1]
# y_lims = [0.85, 1]
# y_lims = None
# y_lims = [0.6, 1]

def getEfficiciencyHist(num_binned, den_binned):
    """
       getEfficiciencyHist creates a binned histogram of the ratio of num_binned and den_binned
       and uses a Clopper-Pearson confidence interval to find uncertainties.

       NOTE: num_binned should be a strict subset of den_binned.

       NOTE: efficiency_binned_err[0] is lower error bar and efficiency_binned_err[1] is upper error bar

       INPUT:
             num_binned - TYPE: numpy array-like
             den_binned - TYPE: numpy array-like
       OUTPUT:
             efficiency_binned - TYPE: numpy array-like
             efficiency_binned_err - TYPE: [numpy array-like, numpy array-like]
       
    """
    # Initializing binned data
    efficiency_binned = np.array([])
    efficiency_binned_err = [np.array([]), np.array([])]

    # Iterating through each bin 
    for i in range(0, len(den_binned)):
        # Catching division by 0 error
        if(den_binned[i] == 0):
            efficiency_binned = np.append(efficiency_binned, 0)
            efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [0])
            efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [0])
            continue

        # Filling efficiency bins
        efficiency_binned = np.append(efficiency_binned, [num_binned[i]/den_binned[i]])

        # Calculating Clopper-Pearson confidence interval
        nsuccess = num_binned[i]
        ntrial = den_binned[i]
        conf = 95.0
    
        if nsuccess == 0:
            alpha = 1 - conf / 100
            plo = 0.
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess + 1, ntrial - nsuccess)
        elif nsuccess == ntrial:
            alpha = 1 - conf / 100
            plo = scipy.stats.beta.ppf(alpha, nsuccess, ntrial - nsuccess + 1)
            phi = 1.
        else:
            alpha = 0.5 * (1 - conf / 100)
            plo = scipy.stats.beta.ppf(alpha, nsuccess + 1, ntrial - nsuccess)
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess, ntrial - nsuccess)

        # Filling efficiency error bins
        efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [(efficiency_binned[i] - plo)])
        efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [(phi - efficiency_binned[i])])# - efficiency_binned[i]])

    return efficiency_binned, efficiency_binned_err

def theoretical_efficiency(pt, a, b, c, d):
    # return (1/2) * (1 + erf((pt - a) / (b * np.sqrt(2))))
    return (1/2) * (1 + erf((pt - a) / ((b * (pt ** c) + d) * np.sqrt(2))))
    # return (1/2) * (1 + erf(
    #     ((1 - pt) / a) / (b * np.sqrt(2))
    # ))
    # return a * pt + b

def fit_efficiency(pt_bins, efficiency, efficiency_err):
    x = pt_bins[:-1] + np.diff(pt_bins) / 2
    bounds = ([0,0,0,0],[1000,3,3,3])
    popt, pcov = curve_fit(theoretical_efficiency, xdata=x, ydata=efficiency, bounds=bounds, maxfev=100000)
    return popt

def get_efficiency(gen_pt, predicted_pt, pt_bins, pt_cut):
    passing_muons_per_gen_pt = [np.sum(predicted_pt[np.logical_and(gen_pt > pt_bins[i], gen_pt < pt_bins[i+1])] > pt_cut) for i in range(len(pt_bins) - 1)]
    GEN_pt_binned, _ = np.histogram(gen_pt, bins=bins)
    efficiency, efficiency_err = getEfficiciencyHist(passing_muons_per_gen_pt, GEN_pt_binned)

    return efficiency, efficiency_err

def get_rate_zerobias(pt, nevents, pt_cut):
    return (np.sum(pt > pt_cut) / nevents) * 40e9

def get_rate(efficiency, bins, pt_cut):
    a = 9.46308e+05
    b = 0.76018
    c = -3.66
    def muon_frequency(pt):
        A = 9.463e5
        b = -0.76018
        c = -3.66

        return A * ((pt - b) ** c)
    
    passing_muon_count = np.zeros(len(efficiency))

    for i in range(len(bins) - 1):
        passing_muon_count[i], _ = quad(muon_frequency, bins[i], bins[i+1])
    
    return passing_muon_count * efficiency

def binary_classifier(scores, gen_pt, pt_cut):
    fig, ax = plt.subplots(1)

    score_bins = np.linspace(0, 1, 20)

    mask = gen_pt < pt_cut
    counts, _ = np.histogram(scores[mask], bins=score_bins)
    ax.stairs(counts / np.sum(counts), score_bins, label="pt_gen < " + str(pt_cut))    
    mask = gen_pt > pt_cut
    counts, _ = np.histogram(scores[mask], bins=score_bins)
    ax.stairs(counts / np.sum(counts), score_bins, label="pt_gen > " + str(pt_cut))

    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig, ax

def reweight(efficiencies, efficiency_errors, weights):
    weights /= np.sum(weights)
    weighted_efficiencies = efficiencies * weights[:, np.newaxis]
    weighted_efficiency_errors = efficiency_errors * weights[:, np.newaxis, np.newaxis]
    return np.sum(weighted_efficiencies, axis=0), np.sum(weighted_efficiency_errors, axis=0)

def resolution(predicted_pts, gen_pts, labels, resolution_bins = np.linspace(-2, 2, 51)):
    fig, ax = plt.subplots(1)
    for i in range(len(predicted_pts)):
        res = (predicted_pts[i] / gen_pts[i]) - 1
        counts, bins = np.histogram(res, bins = resolution_bins)
        a = ax.stairs(counts / np.sum(counts), bins, label=labels[i])
        ax.vlines(np.mean(res), 0, 0.05, linestyle="dashed", colors=a.get_edgecolor())
        ax.scatter(np.mean(res), 0.05, color=a.get_edgecolor())

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, ax

def split_low_high(efficiencies, efficiency_errors, labels, low_xlims=[0, 50], high_xlims=[0, 1000]):
    fig, [low_pt, high_pt] = plt.subplots(2,1)
    x = bins[:-1] + np.diff(bins) / 2

    for i in range(len(efficiencies)):
        efficiency = efficiencies[i]
        efficiency_err = efficiency_errors[i]
        stairs_plot = low_pt.scatter(x, efficiency, label=labels[i], s=1)
        color = stairs_plot.get_edgecolor()
        high_pt.scatter(x, efficiency, label=labels[i], color=color, s=1)

        low_pt.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
        high_pt.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)


    low_pt.set_ylabel("Efficiency", fontsize=12)
    # low_pt.set_xlabel(r"$p_T$", fontsize=12)
    high_pt.set_ylabel("Efficiency", fontsize=12)
    high_pt.set_xlabel(r"$p_T$ (GeV)", fontsize=12)
    fig.align_ylabels([low_pt, high_pt])

    low_pt.set_xlim(low_xlims)
    low_pt.axvline(pt_cut, color="black", linestyle="dashed")
    # low_pt.axhline(0.9, color="black", linestyle="dashed")

    if y_lims != None:
        high_pt.set_ylim(y_lims)
    else:
        high_pt.set_ylim(.95 * efficiencies[0][-1], 1)
    high_pt.set_xlim(high_xlims)

    low_pt.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, markerscale=5)

    fig.suptitle(fig_name)
    return fig, [low_pt, high_pt]

def resolution_3d(gen_pt, predicted_pt, gen_pt_bins):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bins = np.linspace(-2, 2, 50)
    resolution_distributions = np.zeros((len(gen_pt_bins) - 1, len(bins) - 1))

    for i in range(len(gen_pt_bins) - 1):
        mask = np.logical_and(gen_pt > gen_pt_bins[i], gen_pt < gen_pt_bins[i + 1])
        counts, _ = np.histogram((gen_pt[mask] - predicted_pt[mask]) / gen_pt[mask], bins=bins)
        resolution_distributions[i, :] = counts / np.sum(counts)

    x = bins[:-1] + np.diff(bins) / 2
    resolution_line_positions = gen_pt_bins[:-1] + np.diff(gen_pt_bins) / 2

    for i in range(resolution_distributions.shape[0]):
        y = resolution_line_positions[i] * np.ones_like(x)  # Z-axis positions
        z = resolution_distributions[i, :]  # Y-axis (distribution values)
        
        ax.plot(x, y, z, label=f'gen_pt_bin {i+1}')

    
    # Set labels
    ax.set_xlabel('Resolution (gen_pt - predicted_pt) / gen_pt')
    ax.set_ylabel('gen_pt_bins')
    ax.set_zlabel('Frequency (Normalized)')

    return fig, ax

def get_scale_factor(gen_pt, predicted_pt, pt_cut):
    bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
        20,22,24,26,28,30,32,34,36,38,40,42,
        44,46,48,50,60,70,80,90,100,150,200,
        250,300,400,500,600,700,800,900,1000]
    efficiency, efficiency_err = get_efficiency(gen_pt, predicted_pt, bins, pt_cut)
    popt = fit_efficiency(bins, efficiency, efficiency_err)
    
    def target_func(x):
        return theoretical_efficiency(x, *popt) - 0.9
    
    pt_90 = fsolve(target_func, pt_cut)[0]
    return pt_90 / pt_cut

def get_scaled_prediction(gen_pt, predicted_pt, scale_up_to=40):
    scale_bins = np.arange(1, scale_up_to)
    scale_pt_cuts = scale_bins[:-1] + np.diff(scale_bins) / 2
    scale_factors = np.zeros(len(scale_pt_cuts))

    scaled_prediction = np.copy(predicted_pt)

    for i in range(len(scale_pt_cuts)):
        scale_factor = get_scale_factor(gen_pt, predicted_pt, scale_pt_cuts[i])
        mask = np.logical_and(predicted_pt >= scale_bins[i], predicted_pt < scale_bins[i + 1])
        scaled_prediction[mask] = scaled_prediction[mask] * scale_factor
        scale_factors[i] = scale_factor
    
    return scaled_prediction, scale_factors

def with_ratio(xlims):
    # Ratio will be wrt the first path
    fig, [efficiency_ax, ratio_ax] = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(6, 6))

    x = bins[:-1] + np.diff(bins) / 2
    first_path_efficiency = None  # To store the efficiency from the first path
    first_path_efficiency_err = None

    maxr = 0

    for j, path in enumerate(paths):    
        with open(path, 'rb') as file:
            prediction_dict = pickle.load(file)

        print(prediction_dict["training_features"])

        predicted_pt = prediction_dict['predicted_pt']
        gen_pt = prediction_dict['gen_data'][:, prediction_dict['gen_features'] == "gen_pt"].squeeze()

        GEN_pt_binned, _ = np.histogram(gen_pt, bins=bins)

        # Count the number of muons in a given GEN_pt bin that were assigned a pT greater than the threshold by the BDT
        a = [np.sum(predicted_pt[np.logical_and(gen_pt > bins[i], gen_pt < bins[i+1])] > pt_cut) for i in range(len(bins) - 1)]

        efficiency_binned, efficiency_binned_err = getEfficiciencyHist(a, GEN_pt_binned)

        efficiency = np.nan_to_num(a / GEN_pt_binned)

        # Store the efficiency of the first path to compare later
        if j == 0:
            first_path_efficiency = efficiency
            first_path_efficiency_err = efficiency_binned_err


        stairs_plot = efficiency_ax.scatter(x, efficiency, label=labels[j], s=1)
        color = stairs_plot.get_edgecolor()
        efficiency_ax.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)],
                        efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                        linestyle="", marker=".", markersize=5, elinewidth = .5, color=color)

        # Calculate and plot the ratio for paths other than the first
        if j > 0:
            # ratio = np.nan_to_num((efficiency - first_path_efficiency) / first_path_efficiency)  # Prevent NaNs in case of 0/0
            ratio = np.nan_to_num(efficiency / first_path_efficiency, nan=0.0, posinf=0.0, neginf=0.0)  # Prevent NaNs in case of 0/0
            ratio_ax.scatter(x, ratio, s=1, color=color)

            ratio_err = np.nan_to_num(
                ratio * np.sqrt(
                    (np.array(efficiency_binned_err) / np.array(efficiency_binned))**2 + 
                    (np.array(first_path_efficiency_err) / np.array(first_path_efficiency))**2
                ),
                nan=0.0, posinf=0.0, neginf=0.0
            )

            maxr = max(maxr, np.max(ratio))

            ratio_ax.errorbar(
                x, ratio, yerr=ratio_err, xerr=np.diff(bins) / 2,
                linestyle="", marker=".", markersize=5, elinewidth=0.5, color=color
            )

    efficiency_ax.set_ylabel("Efficiency")
    efficiency_ax.xaxis.set_visible(False)
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.set_xlabel(r"$p_T$")

    ratio_ax.set_xlim(xlims)
    efficiency_ax.set_xlim(xlims)

    efficiency_ax.axvline(pt_cut, color="black", linestyle="dashed")
    ratio_ax.axhline(1, color="gray", linestyle="dashed")
    ratio_ax.axvline(pt_cut, color="black", linestyle="dashed")
    efficiency_ax.set_ylim(y_lims)
    efficiency_ax.legend()

    fig.subplots_adjust(hspace=0.01)
    fig.suptitle(fig_name)
    return fig

def get_prediction_dict(path):
    with open(path, 'rb') as file:
            prediction_dict = pickle.load(file)
    predicted_pt = prediction_dict['predicted_pt']
    gen_pt = prediction_dict['gen_data'][:, prediction_dict['gen_features'] == "gen_pt"].squeeze()
    return prediction_dict, predicted_pt, gen_pt

# with open(paths[1], "rb") as file:
#     prediction_dict = pickle.load(file)
#     testing_dataset = get_by_name(prediction_dict["testing_dataset"], config.WRAPPER_DICT_NAME)['training_data_builder']
#     print(testing_dataset.get_features(["bend_1", "dPhi_12"]))
#     print(prediction_dict["predicted_pt"])
#     print(np.sum(prediction_dict["predicted_pt"] > pt_cut))
#     print(len(testing_dataset.data))
#     rate = get_rate_zerobias(prediction_dict["predicted_pt"], len(testing_dataset.data), pt_cut=22)
#     print(f"Rate: {rate} Hz")


efficiencies = np.zeros((len(paths), len(bins) - 1))
efficiency_errors = np.zeros((len(paths), 2, len(bins) - 1))

predicted_pts = []
gen_pts = []

for i, path in enumerate(paths):
    prediction_dict, predicted_pt, gen_pt = get_prediction_dict(path)
    if eta_bounds != None:
        eta = prediction_dict["gen_data"][:, prediction_dict["gen_features"] == "gen_eta"].flatten()
        mask = np.logical_and(eta > eta_bounds[0], eta < eta_bounds[1])
        gen_pt = gen_pt[mask]
        predicted_pt = predicted_pt[mask]
    
    predicted_pts.append(predicted_pt)
    gen_pts.append(gen_pt)
    efficiencies[i, :], efficiency_errors[i, :, :] = get_efficiency(gen_pt, predicted_pt, bins, pt_cut)

# weights = np.array([mode_weights[15], mode_weights[14], mode_weights[13], mode_weights[11]])

# figure, ax = resolution(predicted_pts, gen_pts, labels)
figure, axs = split_low_high(efficiencies, efficiency_errors, labels)

def unique_name(filename, directory="."):
    # Set the initial file path
    base_filepath = os.path.join(directory, filename + ".png")
    filepath = base_filepath
    counter = 1
    # Check if the file exists, and if it does, append a number to make the name unique
    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{filename}_{counter}.png")
        counter += 1
    
    return filepath

os.makedirs(os.path.join(config.FIGURES_DIRECTORY, fig_dir), exist_ok=True)
plt.savefig(unique_name(f"efficiency_{fig_name}", directory = os.path.join(config.FIGURES_DIRECTORY, fig_dir)), dpi=300)

figure, ax = resolution(predicted_pts, gen_pts, labels)
plt.savefig(unique_name(f"resolution_{fig_name}", directory = os.path.join(config.FIGURES_DIRECTORY, fig_dir)), dpi=300)