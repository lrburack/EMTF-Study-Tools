import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_xlim([-1.5, 1.5])
    ax.legend()
    return fig, ax

def binned_resolution(predicted_pt, gen_pt, pt_bins=np.linspace(0, 50, 51), resolution_bins=np.linspace(-2, 2, 19)):
    plot_values = np.zeros((len(pt_bins) - 1, len(resolution_bins) - 1))

    for i in range(len(pt_bins) - 1):
        mask = (gen_pt > pt_bins[i]) & (gen_pt < pt_bins[i + 1])
        res = (predicted_pt[mask] / gen_pt[mask]) - 1
        counts, _ = np.histogram(res, resolution_bins)
        plot_values[i, :] = counts / np.sum(counts)
    
    fig, ax = plt.subplots(1, figsize=(12,4))

    ax.set_xlabel(r"$p_T$")
    ax.set_ylabel("Resolution")
    mesh = ax.pcolormesh(pt_bins, resolution_bins, plot_values.T, shading='flat')

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.2, fraction=0.1)
    cbar.set_label(r'Normalized by $p_T$ bin')

    return fig, ax


def binned_resolution_with_whisker(predicted_pt, gen_pt, pt_bins=np.linspace(0, 50, 51), resolution_bins=np.linspace(-2, 2, 19)):
    plot_values = np.zeros((len(pt_bins) - 1, len(resolution_bins) - 1))
    means = []
    std_devs = []
    pt_bin_centers = []

    for i in range(len(pt_bins) - 1):
        mask = (gen_pt > pt_bins[i]) & (gen_pt < pt_bins[i + 1])
        res = (predicted_pt[mask] / gen_pt[mask]) - 1
        if len(res) > 0:  # Avoid empty bins
            means.append(np.mean(res))
            std_devs.append(np.std(res))
            pt_bin_centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)
        else:
            means.append(np.nan)
            std_devs.append(0)
            pt_bin_centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)

        counts, _ = np.histogram(res, resolution_bins)
        plot_values[i, :] = counts / np.sum(counts) if np.sum(counts) > 0 else 0

    fig, ax = plt.subplots(1, figsize=(12, 4))
    mesh = ax.pcolormesh(pt_bins, resolution_bins, plot_values.T, shading='flat')

    # Overlay mean and std deviation whiskers
    ax.errorbar(pt_bin_centers, means, yerr=std_devs, fmt='o', color='red', ecolor='red',
                elinewidth=1, capsize=2, label='Mean ± 1 Std Dev')

    ax.set_xlabel(r"$p_T$")
    ax.set_ylabel("Resolution")
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.2, fraction=0.1)
    cbar.set_label(r'Normalized by $p_T$ bin')

    ax.legend()
    return fig, ax


def binned_spread(predicted_pt, gen_pt, pt_bins=np.linspace(0, 50, 51), spread_bins=np.linspace(-10, 10, 21)):
    plot_values = np.zeros((len(pt_bins) - 1, len(spread_bins) - 1))

    for i in range(len(pt_bins) - 1):
        mask = (gen_pt > pt_bins[i]) & (gen_pt < pt_bins[i + 1])
        spread = predicted_pt[mask] - gen_pt[mask]
        counts, _ = np.histogram(spread, spread_bins)
        plot_values[i, :] = counts / np.sum(counts)
    
    fig, ax = plt.subplots(1, figsize=(12,4))

    ax.set_xlabel(r"$p_T$")
    ax.set_ylabel("Spread (pred - true)")
    mesh = ax.pcolormesh(pt_bins, spread_bins, plot_values.T, shading='flat')

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.2, fraction=0.1)
    cbar.set_label(r'Normalized by $p_T$ bin')

    return fig, ax


def binned_spread_with_whisker(predicted_pt, gen_pt, pt_bins=np.linspace(0, 50, 51), spread_bins=np.linspace(-20, 20, 21)):
    plot_values = np.zeros((len(pt_bins) - 1, len(spread_bins) - 1))
    means = []
    std_devs = []
    pt_bin_centers = []

    for i in range(len(pt_bins) - 1):
        mask = (gen_pt > pt_bins[i]) & (gen_pt < pt_bins[i + 1])
        spread = predicted_pt[mask] - gen_pt[mask]
        if len(spread) > 0:  # Avoid empty bins
            means.append(np.mean(spread))
            std_devs.append(np.std(spread))
            pt_bin_centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)
        else:
            means.append(np.nan)
            std_devs.append(0)
            pt_bin_centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)

        counts, _ = np.histogram(spread, spread_bins)
        plot_values[i, :] = counts / np.sum(counts) if np.sum(counts) > 0 else 0

    fig, ax = plt.subplots(1, figsize=(12, 4))
    mesh = ax.pcolormesh(pt_bins, spread_bins, plot_values.T, shading='flat')

    # Overlay mean and std deviation whiskers
    ax.errorbar(pt_bin_centers, means, yerr=std_devs, fmt='o', color='red', ecolor='red',
                elinewidth=1, capsize=2, label='Mean ± 1 Std Dev')

    ax.set_xlabel(r"$p_T$")
    ax.set_ylabel("Spread (pred - true)")
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.2, fraction=0.1)
    cbar.set_label(r'Normalized by $p_T$ bin')

    ax.legend()
    return fig, ax