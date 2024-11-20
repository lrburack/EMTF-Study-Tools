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