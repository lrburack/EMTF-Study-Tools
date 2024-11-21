import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(predicted_pt, true_pt, pt_cut):
    # Define the truth and predicted binary classifications
    truth = true_pt > pt_cut
    predicted = predicted_pt > pt_cut

    # Compute confusion matrix values
    true_positive = np.sum((predicted == 1) & (truth == 1))
    false_positive = np.sum((predicted == 1) & (truth == 0))
    true_negative = np.sum((predicted == 0) & (truth == 0))
    false_negative = np.sum((predicted == 0) & (truth == 1))

    # Confusion matrix as a 2x2 array
    confusion = np.array([[true_negative, false_positive],
                          [false_negative, true_positive]])

    # Create plot
    fig, ax = plt.subplots(1, figsize=(3,3))
    cax = ax.imshow(confusion, cmap='Blues', interpolation='nearest')

    # Add labels and colorbar
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([r"$p_T$" + f" < {pt_cut}GeV", r"$p_T$" + f" > {pt_cut}GeV"])
    ax.set_yticklabels([r"$p_T$" + f" < {pt_cut}GeV", r"$p_T$" + f" > {pt_cut}GeV"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")

    # Annotate cells with values
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center', color='black')

    fig.tight_layout()
    return fig, ax

def parity_plot(predicted_pt, true_pt, bins=np.linspace(0, 50, 51)):
    fig, ax = plt.subplots()
    
    hist, xedges, yedges = np.histogram2d(true_pt, predicted_pt, bins=bins)
    hist_normalized = hist / np.sum(hist, axis=0, keepdims=True)
    hist_normalized = np.nan_to_num(hist_normalized)  # Replace NaNs with 0
    
    mesh = ax.pcolormesh(xedges, yedges, hist_normalized.T, shading='flat', cmap='Blues')
    
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Normalized by Column')
    
    ax.set_xlabel("True $p_T$")
    ax.set_ylabel("Predicted $p_T$")
    
    return fig, ax

