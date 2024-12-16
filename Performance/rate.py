import numpy as np
import pickle
import os
import config
import matplotlib.pyplot as plt
from Performance.scale_factor import current_EMTF_scale_pt, generic_scale_pt

# BX freq in kHz * kHz to Hz * # of filled bunches in the LHC
COLLISION_FREQ = 11.245 * 1000 * 2340

def lm_pt(pt, event_correspondance):
    # We can have multiple tracks from the same event so we need to find the one with the highest pt
    sort_indices = np.lexsort((-pt, event_correspondance))
    sorted_events = event_correspondance[sort_indices]
    sorted_pts = pt[sort_indices]
    
    # Find unique events and their indices (keeping only the first occurrence)
    unique_events, unique_indices = np.unique(sorted_events, return_index=True)
    
    # Select the leading pt values for each unique event
    leading_muon_pt = sorted_pts[unique_indices]

    return leading_muon_pt, sorted_events[unique_indices]

def get_rate_zerobias_NTuples(paths, labels, pt_cut, pt_scale_fxn=None):
    # These paths should be paths to wrapper dicts
    nevents = 0
    pt = np.empty(len(paths), dtype=object)
    event_correspondance = np.empty(len(paths), dtype=object)

    for i in range(len(paths)):
        with open(paths[i], "rb") as file:
            dataset = pickle.load(file)["dataset"]

        pt[i] = dataset.get_features(["emtfTrack_pt"])
        event_correspondance[i] = dataset.event_correspondance

        if i == 0:
            nevents = dataset.events_processed
        elif dataset.events_processed != nevents:
            raise ValueError("All datasets must have the same number of events")
    
    pt = np.concatenate(pt)
    event_correspondance = np.concatenate(event_correspondance)

    leading_muon_pt, events = lm_pt(pt, event_correspondance)

    # print(leading_muon_pt )
    return get_rate(leading_muon_pt, nevents, pt_cut)

def get_rate_zerobias_prediction(paths, pt_cut, pt_scale_fxn=current_EMTF_scale_pt):
    leading_muon_pt, nevents = get_leading_muon_pt(paths)
    if pt_scale_fxn != None:
        leading_muon_pt = pt_scale_fxn(leading_muon_pt) * leading_muon_pt
    return get_rate(leading_muon_pt, nevents, pt_cut)

def get_leading_muon_pt(paths, scaling_constants = None):
    if isinstance(paths, str):
        paths = [paths]

    nevents = 0
    pt = np.empty(len(paths), dtype=object)
    event_correspondance = np.empty(len(paths), dtype=object)

    for i in range(len(paths)):
        with open(paths[i], "rb") as file:
            prediction_dict = pickle.load(file)
        dataset = prediction_dict["testing_dataset"]

        pt[i] = prediction_dict["predicted_pt"]
        event_correspondance[i] = dataset.event_correspondance

        if scaling_constants is not None:
            pt[i] = pt[i] * generic_scale_pt(pt[i], *scaling_constants[i])

        if i == 0:
            nevents = dataset.events_processed
        elif dataset.events_processed != nevents and dataset.events_processed != nevents + 1:
            raise ValueError(f"All datasets must have the same number of events. Expected {nevents}, but {paths[i]} had {dataset.events_processed}")
    
    pt = np.concatenate(pt)
    event_correspondance = np.concatenate(event_correspondance)

    leading_muon_pt, events = lm_pt(pt, event_correspondance)
    return leading_muon_pt, nevents

def get_rate(leading_muon_pt, nevents, pt_cut):
    return (np.sum(leading_muon_pt > pt_cut) / nevents) * COLLISION_FREQ



# def rate_plot(predictions, pt_cuts=np.linspace(0, 50, 51), pt_cut_to_annotate=None):
#     fig, ax = plt.subplots(1)

#     labels = predictions.keys()
    
#     for i, label in enumerate(labels):
#         paths = [os.path.join(config.STUDY_DIRECTORY, path) for path in predictions[label]]
#         leading_muon_pt, nevents = get_leading_muon_pt(paths)

#         passing_muons = np.zeros(len(pt_cuts))
#         for j in range(len(pt_cuts)):
#             passing_muons[j] = np.sum(leading_muon_pt > pt_cuts[j])
        
#         rates = (passing_muons / nevents) * COLLISION_FREQ / 1000
#         a = ax.plot(pt_cuts, rates, label=label)

#         if pt_cut_to_annotate:
#             color = a[0].get_color()
#             rate_at_pt_cut = (np.sum(leading_muon_pt > pt_cut_to_annotate) / nevents) * COLLISION_FREQ / 1000
#             text_pos = [pt_cut_to_annotate + 4 + (4 * i), rate_at_pt_cut * 2 ** (len(labels) - i)]

#             ax.annotate(
#                 "",
#                 xy=(pt_cut_to_annotate, rate_at_pt_cut), 
#                 xytext=(text_pos),
#                 arrowprops=dict(arrowstyle="-", linestyle="dashed", color=color, lw=1),
#                 zorder=50 + i  # Arrow z-order
#             )

#             # Draw the text separately with a higher z-order
#             ax.text(
#                 text_pos[0], 
#                 text_pos[1], 
#                 f"{1000 * rate_at_pt_cut:.0f} Hz", 
#                 fontsize=12, 
#                 ha='left', 
#                 zorder=100,  # Ensure text is on top
#                 color=color  # Match text color
#             )

#             ax.plot(pt_cut_to_annotate, rate_at_pt_cut, 'o', color=color, markersize=4, label='_nolegend_', zorder=50+i)

#     if pt_cut_to_annotate:
#         ax.axvline(x=pt_cut_to_annotate, linestyle="dashed", color="black")

#     ax.set_xlabel(r"$p_T$ threshold (GeV)")
#     ax.set_ylabel("Rate (kHz)")
#     ax.set_yscale("log")

#     ax.legend()

#     return fig, ax

# def rate_plot_with_ratio(predictions, pt_cuts=np.linspace(0, 50, 51), pt_cut_to_annotate=None):
#     fig, [rate_ax, ratio_ax] = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}, figsize=(6, 6))

#     labels = predictions.keys()

#     control_rate = None

#     for i, label in enumerate(labels):
#         paths = [os.path.join(config.STUDY_DIRECTORY, path) for path in predictions[label]]
#         leading_muon_pt, nevents = get_leading_muon_pt(paths)

#         passing_muons = np.zeros(len(pt_cuts))
#         for j in range(len(pt_cuts)):
#             passing_muons[j] = np.sum(leading_muon_pt > pt_cuts[j])
        
#         rates = (passing_muons / nevents) * COLLISION_FREQ / 1000
#         plot = rate_ax.plot(pt_cuts, rates, label=label)

#         if i == 0:
#             control_rate = rates
#         else:
#             ratio_ax.plot(pt_cuts, np.nan_to_num(rates / control_rate), color=plot[0].get_color())
        
#         if pt_cut_to_annotate:
#             color = plot[0].get_color()
#             rate_at_pt_cut = (np.sum(leading_muon_pt > pt_cut_to_annotate) / nevents) * COLLISION_FREQ / 1000
#             text_pos = [pt_cut_to_annotate + 4 + (4 * i), rate_at_pt_cut * 2 ** (len(labels) - i)]

#             rate_ax.annotate(
#                 "",
#                 xy=(pt_cut_to_annotate, rate_at_pt_cut), 
#                 xytext=(text_pos),
#                 arrowprops=dict(arrowstyle="-", linestyle="dashed", color=color, lw=1),
#                 zorder=50 + i  # Arrow z-order
#             )

#             # Draw the text separately with a higher z-order
#             rate_ax.text(
#                 text_pos[0], 
#                 text_pos[1], 
#                 f"{1000 * rate_at_pt_cut:.0f} Hz", 
#                 fontsize=12, 
#                 ha='left', 
#                 zorder=100,  # Ensure text is on top
#                 color=color  # Match text color
#             )

#             rate_ax.plot(pt_cut_to_annotate, rate_at_pt_cut, 'o', color=color, markersize=4, label='_nolegend_', zorder=50+i)

#     ratio_ax.set_xlabel(r"$p_T$ threshold (GeV)")
#     ratio_ax.set_ylabel("Ratio")
#     rate_ax.set_ylabel("Rate (kHz)")
#     rate_ax.set_yscale("log")
#     rate_ax.xaxis.set_visible(False)

#     ratio_ax.axhline(y=1, linestyle="dashed", color="black", zorder=-10)
#     if pt_cut_to_annotate:
#         rate_ax.axvline(x=pt_cut_to_annotate, linestyle="dashed", color="black")

#     rate_ax.legend()

#     return fig, [rate_ax, ratio_ax]


def rate_plot(leading_muon_pts, labels, pt_cuts=np.linspace(0, 50, 51), pt_cut_to_annotate=None):
    fig, ax = plt.subplots(1)
    
    for i, label in enumerate(labels):
        leading_muon_pt = leading_muon_pts[i]
        nevents = len(leading_muon_pt)

        passing_muons = np.zeros(len(pt_cuts))
        for j in range(len(pt_cuts)):
            passing_muons[j] = np.sum(leading_muon_pt > pt_cuts[j])
        
        rates = (passing_muons / nevents) * COLLISION_FREQ / 1000
        a = ax.plot(pt_cuts, rates, label=label)

        if pt_cut_to_annotate:
            color = a[0].get_color()
            rate_at_pt_cut = (np.sum(leading_muon_pt > pt_cut_to_annotate) / nevents) * COLLISION_FREQ / 1000
            text_pos = [pt_cut_to_annotate + 4 + (4 * i), rate_at_pt_cut * 2 ** (len(labels) - i)]

            ax.annotate(
                "",
                xy=(pt_cut_to_annotate, rate_at_pt_cut), 
                xytext=(text_pos),
                arrowprops=dict(arrowstyle="-", linestyle="dashed", color=color, lw=1),
                zorder=50 + i  # Arrow z-order
            )

            # Draw the text separately with a higher z-order
            ax.text(
                text_pos[0], 
                text_pos[1], 
                f"{1000 * rate_at_pt_cut:.0f} Hz", 
                fontsize=12, 
                ha='left', 
                zorder=100,  # Ensure text is on top
                color=color  # Match text color
            )

            ax.plot(pt_cut_to_annotate, rate_at_pt_cut, 'o', color=color, markersize=4, label='_nolegend_', zorder=50+i)

    if pt_cut_to_annotate:
        ax.axvline(x=pt_cut_to_annotate, linestyle="dashed", color="black")

    ax.set_xlabel(r"$p_T$ threshold (GeV)")
    ax.set_ylabel("Rate (kHz)")
    ax.set_yscale("log")

    ax.legend()

    return fig, ax

def rate_plot_with_ratio(leading_muon_pts, labels, pt_cuts=np.linspace(0, 50, 51), pt_cut_to_annotate=None):
    fig, [rate_ax, ratio_ax] = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}, figsize=(6, 6))

    control_rate = None

    for i, label in enumerate(labels):
        leading_muon_pt = leading_muon_pts[i]
        nevents = len(leading_muon_pt)

        passing_muons = np.zeros(len(pt_cuts))
        for j in range(len(pt_cuts)):
            passing_muons[j] = np.sum(leading_muon_pt > pt_cuts[j])
        
        rates = (passing_muons / nevents) * COLLISION_FREQ / 1000
        plot = rate_ax.plot(pt_cuts, rates, label=label)

        if i == 0:
            control_rate = rates
        else:
            ratio_ax.plot(pt_cuts, np.nan_to_num(rates / control_rate), color=plot[0].get_color())
        
        if pt_cut_to_annotate:
            color = plot[0].get_color()
            rate_at_pt_cut = (np.sum(leading_muon_pt > pt_cut_to_annotate) / nevents) * COLLISION_FREQ / 1000
            text_pos = [pt_cut_to_annotate + 4 + (4 * i), rate_at_pt_cut * 2 ** (len(labels) - i)]

            rate_ax.annotate(
                "",
                xy=(pt_cut_to_annotate, rate_at_pt_cut), 
                xytext=(text_pos),
                arrowprops=dict(arrowstyle="-", linestyle="dashed", color=color, lw=1),
                zorder=50 + i  # Arrow z-order
            )

            # Draw the text separately with a higher z-order
            rate_ax.text(
                text_pos[0], 
                text_pos[1], 
                f"{1000 * rate_at_pt_cut:.0f} Hz", 
                fontsize=12, 
                ha='left', 
                zorder=100,  # Ensure text is on top
                color=color  # Match text color
            )

            rate_ax.plot(pt_cut_to_annotate, rate_at_pt_cut, 'o', color=color, markersize=4, label='_nolegend_', zorder=50+i)

    ratio_ax.set_xlabel(r"$p_T$ threshold (GeV)")
    ratio_ax.set_ylabel("Ratio")
    rate_ax.set_ylabel("Rate (kHz)")
    rate_ax.set_yscale("log")
    rate_ax.xaxis.set_visible(False)

    ratio_ax.axhline(y=1, linestyle="dashed", color="black", zorder=-10)
    if pt_cut_to_annotate:
        rate_ax.axvline(x=pt_cut_to_annotate, linestyle="dashed", color="black")

    rate_ax.legend()

    return fig, [rate_ax, ratio_ax]