import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

def pt_distribution(pts, labels, bins = np.linspace(0, 50, 26)):
    fig, ax = plt.subplots(1)

    for i in range(len(pts)):
        counts, _ = np.histogram(pts[i], bins=bins)
        ax.stairs(counts / np.sum(counts), bins, label=labels[i])
    
    ax.set_xlabel(r"$p_T$ (GeV)")
    ax.legend()
    # ax.set_ylabel()

    return fig, ax

def model_compare_all_pT_distribution(pts, true_pts, labels, pt_cut=22, pt_bins=np.linspace(0, 50, 26)):
    # Must all be made from the same dataset (true_pts[:] must be identical)
    fig, ax = plt.subplots(1)

    ax.set_title(f"Fraction of muons that pass the {pt_cut} GeV threshold")
    ax.set_ylabel("Fraction per pT bin")
    ax.set_xlabel("GEN pT (GeV)")

    muons_per_pt_bin, _ = np.histogram(np.clip(true_pts[0], pt_bins[0], pt_bins[-1]), bins=pt_bins)
    # muons_per_pt_bin, _ = np.histogram(true_pts[0], bins=pt_bins)

    x = pt_bins[:-1] + np.diff(pt_bins) / 2

    ymax = 0

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            mask = (pts[i] > pt_cut) & (pts[j] < pt_cut)
            label = f"Passes '{labels[i]}' but not '{labels[j]}'"
            counts, _ = np.histogram(np.clip(true_pts[i][mask], pt_bins[0], pt_bins[-1]), bins=pt_bins)
            # counts, _ = np.histogram(true_pts[i][mask], bins=pt_bins)
            ax.plot(x, counts / muons_per_pt_bin, label=label)

            ymax = max(ymax, np.max(np.nan_to_num(counts / muons_per_pt_bin)))

    ax.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)})
    ax.axvline(x=pt_cut, color="black", linestyle="dashed", zorder=-10)
    ax.set_ylim([0, ymax*1.2])

    return fig, ax

def model_compare_control_pT_distribution(control_predicted_pt, new_predicted_pt, true_pt, labels, pt_cut=22, pt_bins=np.linspace(0, 50, 26)):
    # Control and new prediction must be made from the same events
    fig, ax = plt.subplots(1)

    ax.set_title(f"Fraction of muons that pass the {pt_cut} GeV threshold")
    ax.set_ylabel(r"Fraction per $p_T$ bin")
    ax.set_xlabel(r"GEN $p_T$ (GeV)")

    muons_per_pt_bin, _ = np.histogram(np.clip(true_pt, pt_bins[0], pt_bins[-1]), bins=pt_bins)

    x = pt_bins[:-1] + np.diff(pt_bins) / 2

    mask = (new_predicted_pt < pt_cut) & (control_predicted_pt >= pt_cut)
    old_counts, _ = np.histogram(np.clip(true_pt[mask], pt_bins[0], pt_bins[-1]), bins=pt_bins)
    ax.plot(x, old_counts / muons_per_pt_bin, label=r"Pass current $p_T$ assignment only")
    mask = (new_predicted_pt >= pt_cut) & (control_predicted_pt < pt_cut)
    new_counts, _ = np.histogram(np.clip(true_pt[mask], pt_bins[0], pt_bins[-1]), bins=pt_bins)
    ax.plot(x, new_counts / muons_per_pt_bin, label=r"Pass new $p_T$ assignment only")

    ax.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, fontsize=9)
    ax.axvline(x=pt_cut, color="black", linestyle="dashed", zorder=-10)
    ax.set_ylim([0, 1.2 * max(np.max(np.nan_to_num(new_counts / muons_per_pt_bin)), np.max(np.nan_to_num(old_counts / muons_per_pt_bin)))])

    return fig, ax

bins = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000])

def model_compare_split_pT_distribution(pts, true_pts, labels, pt_cut=22, low_pt_bins=np.linspace(0, 50, 26), high_pt_bins=bins):
    # Must all be made from the same dataset (true_pts[:] must be identical)
    # The models will be compared to the first model passed
    fig, [low_pt, high_pt] = plt.subplots(2)

    fig.suptitle(f"Model comparison at {pt_cut} GeV")
    low_pt.set_ylabel("Fraction per pT bin")
    high_pt.set_ylabel("Fraction per pT bin")
    high_pt.set_xlabel("pT (GeV)")

    low_muons_per_pt_bin, _ = np.histogram(true_pts[0], bins=low_pt_bins)
    high_muons_per_pt_bin, _ = np.histogram(true_pts[0], bins=high_pt_bins)

    low_x = low_pt_bins[:-1] + np.diff(low_pt_bins) / 2
    high_x = high_pt_bins[:-1] + np.diff(high_pt_bins) / 2

    for i in range(1, len(labels)):
        mask = (pts[i] > pt_cut) & (pts[0] < pt_cut)
        label = f"Passes '{labels[i]}' but not '{labels[0]}'"
        low_counts, _ = np.histogram(true_pts[i][mask], bins=low_pt_bins)
        low_pt.scatter(low_x, low_counts / low_muons_per_pt_bin, label=label, s=3)
        high_counts, _ = np.histogram(true_pts[i][mask], bins=high_pt_bins)
        high_pt.scatter(high_x, high_counts / high_muons_per_pt_bin, label=label, s=3)
    
    low_pt.legend(handler_map={plt.scatter: HandlerPathCollection(marker_pad=0)}, markerscale=5, bbox_to_anchor=(0.5, 1.2), loc='center')
    low_pt.axvline(x=pt_cut, color="gray", linestyle="dashed")
    high_pt.axvline(x=pt_cut, color="gray", linestyle="dashed")

    return fig, [low_pt, high_pt]