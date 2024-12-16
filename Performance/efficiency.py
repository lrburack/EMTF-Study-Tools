import numpy as np
import scipy.stats
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt

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

def get_efficiency(gen_pt, predicted_pt, pt_bins, pt_cut):
    passing_muons_per_gen_pt = [np.sum(predicted_pt[np.logical_and(gen_pt > pt_bins[i], gen_pt < pt_bins[i+1])] > pt_cut) for i in range(len(pt_bins) - 1)]
    GEN_pt_binned, _ = np.histogram(gen_pt, bins=pt_bins)
    efficiency, efficiency_err = getEfficiciencyHist(passing_muons_per_gen_pt, GEN_pt_binned)

    return efficiency, efficiency_err

def reweight(efficiencies, efficiency_errors, weights):
    weights /= np.sum(weights)
    weighted_efficiencies = efficiencies * weights[:, np.newaxis]
    weighted_efficiency_errors = efficiency_errors * weights[:, np.newaxis, np.newaxis]
    return np.sum(weighted_efficiencies, axis=0), np.sum(weighted_efficiency_errors, axis=0)


bins = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000])

def efficiency_split_low_high(predicted_pts, true_pts, labels, pt_bins=bins, pt_cut=22, low_xlims=[0, 50], high_xlims=[0, 1000]):
    fig, [low_pt, high_pt] = plt.subplots(2,1)
    x = pt_bins[:-1] + np.diff(pt_bins) / 2

    min_high_pt_efficiency = 1

    for i in range(len(predicted_pts)):
        efficiency, efficiency_err = get_efficiency(true_pts[i], predicted_pts[i], pt_bins=pt_bins, pt_cut=pt_cut)
        stairs_plot = low_pt.scatter(x, efficiency, label=labels[i], s=1)
        color = stairs_plot.get_edgecolor()
        high_pt.scatter(x, efficiency, label=labels[i], color=color, s=1)

        low_pt.errorbar([pt_bins[i]+(pt_bins[i+1]-pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(pt_bins[i+1] - pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
        high_pt.errorbar([pt_bins[i]+(pt_bins[i+1]-pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(pt_bins[i+1] - pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
        
        min_high_pt_efficiency = min(min_high_pt_efficiency, efficiency[-1])


    low_pt.set_ylabel("Efficiency", fontsize=12)
    # low_pt.set_xlabel(r"$p_T$", fontsize=12)
    high_pt.set_ylabel("Efficiency", fontsize=12)
    high_pt.set_xlabel(r"GEN $p_T$ (GeV)", fontsize=12)
    fig.align_ylabels([low_pt, high_pt])

    low_pt.set_xlim(low_xlims)
    low_pt.axvline(pt_cut, color="black", linestyle="dashed")
    # low_pt.axhline(0.9, color="black", linestyle="dashed")

    high_pt.set_ylim(.95 * min_high_pt_efficiency, 1)
    high_pt.set_xlim(high_xlims)

    low_pt.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, markerscale=5)

    return fig, [low_pt, high_pt]

def efficiency_with_ratio(predicted_pts, true_pts, labels, pt_cut=22, pt_bins=bins):
    fig, [efficiency_ax, ratio_ax] = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}, figsize=(6, 6))
    x = pt_bins[:-1] + np.diff(pt_bins) / 2

    maxr = 0
    min_high_pt_efficiency = 1
    control_efficiency, control_efficiency_err = get_efficiency(true_pts[0], predicted_pts[0], pt_bins=pt_bins, pt_cut=pt_cut)

    for i in range(len(predicted_pts)):
        efficiency, efficiency_err = get_efficiency(true_pts[i], predicted_pts[i], pt_bins=pt_bins, pt_cut=pt_cut)
        plot = efficiency_ax.scatter(x, efficiency, label=labels[i], s=1)
        color = plot.get_edgecolor()

        efficiency_ax.errorbar([pt_bins[i]+(pt_bins[i+1]-pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(pt_bins[i+1] - pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
        
        min_high_pt_efficiency = min(min_high_pt_efficiency, efficiency[-1])

        if i > 0:
            # ratio = np.nan_to_num((efficiency - first_path_efficiency) / first_path_efficiency)  # Prevent NaNs in case of 0/0
            ratio = np.nan_to_num(efficiency / control_efficiency, nan=0.0, posinf=0.0, neginf=0.0)  # Prevent NaNs in case of 0/0
            ratio_ax.scatter(x, ratio, s=1, color=color)

            ratio_err = np.nan_to_num(
                ratio * np.sqrt(
                    (np.array(efficiency_err) / np.array(efficiency))**2 + 
                    (np.array(control_efficiency_err) / np.array(control_efficiency))**2
                ),
                nan=0.0, posinf=0.0, neginf=0.0
            )

            maxr = max(maxr, np.max(ratio))

            ratio_ax.errorbar(
                x, ratio, yerr=ratio_err, xerr=np.diff(pt_bins) / 2,
                linestyle="", marker=".", markersize=5, elinewidth=0.5, color=color
            )

    efficiency_ax.set_ylabel("Efficiency", fontsize=12)
    # low_pt.set_xlabel(r"$p_T$", fontsize=12)
    ratio_ax.set_ylabel("Ratio", fontsize=12)
    ratio_ax.set_xlabel(r"GEN $p_T$ (GeV)", fontsize=12)
    fig.align_ylabels([efficiency_ax, ratio_ax])

    efficiency_ax.xaxis.set_visible(False)
    efficiency_ax.axvline(pt_cut, color="black", linestyle="dashed")
    ratio_ax.axhline(1, color="black", linestyle="dashed")

    efficiency_ax.set_ylim([.95 * min_high_pt_efficiency, 1])
    ratio_ax.set_ylim([0.8, maxr * 1.2])

    efficiency_ax.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, markerscale=5)

    return fig, [efficiency_ax, ratio_ax]

def efficiency_with_difference(predicted_pts, true_pts, labels, pt_cut=22, pt_bins=bins):
    fig, [efficiency_ax, difference_ax] = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}, figsize=(6, 6))
    x = pt_bins[:-1] + np.diff(pt_bins) / 2

    max_difference = 0
    min_high_pt_efficiency = 1
    control_efficiency, control_efficiency_err = get_efficiency(true_pts[0], predicted_pts[0], pt_bins=pt_bins, pt_cut=pt_cut)

    for i in range(len(predicted_pts)):
        efficiency, efficiency_err = get_efficiency(true_pts[i], predicted_pts[i], pt_bins=pt_bins, pt_cut=pt_cut)
        plot = efficiency_ax.scatter(x, efficiency, label=labels[i], s=1)
        color = plot.get_edgecolor()

        efficiency_ax.errorbar([pt_bins[i]+(pt_bins[i+1]-pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        efficiency, yerr=efficiency_err, xerr=[(pt_bins[i+1] - pt_bins[i])/2 for i in range(0, len(pt_bins)-1)],
                        linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
        
        min_high_pt_efficiency = min(min_high_pt_efficiency, efficiency[-1])

        if i > 0:
            # ratio = np.nan_to_num((efficiency - first_path_efficiency) / first_path_efficiency)  # Prevent NaNs in case of 0/0
            difference = efficiency - control_efficiency  # Prevent NaNs in case of 0/0
            difference_ax.scatter(x, difference, s=1, color=color)

            difference_err = np.nan_to_num(
                np.sqrt(
                    np.array(efficiency_err)**2 + 
                    np.array(control_efficiency_err)**2
                ),
                nan=0.0, posinf=0.0, neginf=0.0
            )

            max_difference = max(max_difference, np.max(difference))

            difference_ax.errorbar(
                x, difference, yerr=difference_err, xerr=np.diff(pt_bins) / 2,
                linestyle="", marker=".", markersize=5, elinewidth=0.5, color=color
            )

    efficiency_ax.set_ylabel("Efficiency", fontsize=12)
    # low_pt.set_xlabel(r"$p_T$", fontsize=12)
    difference_ax.set_ylabel("Difference", fontsize=12)
    difference_ax.set_xlabel(r"GEN $p_T$ (GeV)", fontsize=12)
    fig.align_ylabels([efficiency_ax, difference_ax])

    efficiency_ax.xaxis.set_visible(False)
    efficiency_ax.axvline(pt_cut, color="black", linestyle="dashed")

    efficiency_ax.set_ylim([.95 * min_high_pt_efficiency, 1])
    # difference_ax.set_ylim([0, max(maxr, 2)])
    difference_ax.axhline(0, color="black", linestyle="dashed")

    efficiency_ax.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, markerscale=5)

    return fig, [efficiency_ax, difference_ax]

def annotate_cut_info(ax, cut_text, box_position=None):
    if cut_text == None:
        return

    # Use the provided box_position or default to a sensible value
    x, y = box_position if box_position is not None else (0.5, 0.5)  # Default center

    # Add the text box
    ax.text(
        x, y,
        cut_text,
        va="top" if y > 0.5 else "bottom",
        ha="left" if x < 0.5 else "right",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
        transform=ax.transAxes  # Use axis-relative positioning
    )