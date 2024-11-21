import numpy as np
import pickle
import os
import config
import matplotlib.pyplot as plt
from Performance.scale_factor import current_EMTF_scale_pt

# BX freq in kHz * kHz to Hz * # of filled bunches in the LHC
COLLISION_FREQ = 11.245 * 1000 * 2340

def lm_pt(pt, event_correspondance):
    print("lm_pt...")
    sort_indices = np.lexsort((-pt, event_correspondance))
    sorted_events = event_correspondance[sort_indices]
    sorted_pts = pt[sort_indices]
    
    # Find unique events and their indices (keeping only the first occurrence)
    unique_events, unique_indices = np.unique(sorted_events, return_index=True)
    
    # Select the leading pt values for each unique event
    leading_muon_pt = sorted_pts[unique_indices]
    print("done!")
    return leading_muon_pt

def get_rate_zerobias_NTuples(paths, labels, pt_cut, pt_scale_fxn=None):
    return None

def get_rate_zerobias_prediction(paths, pt_cut, pt_scale_fxn=current_EMTF_scale_pt):
    leading_muon_pt, nevents = get_leading_muon_pt(paths)
    if pt_scale_fxn != None:
        leading_muon_pt = pt_scale_fxn(leading_muon_pt) * leading_muon_pt
    return get_rate(leading_muon_pt, nevents, pt_cut)

def get_leading_muon_pt(paths):
    if isinstance(paths, str):
        paths = [paths]

    nevents = 0
    pt = np.empty(len(paths), dtype=object)
    event_correspondance = np.empty(len(paths), dtype=object)

    for i in range(len(paths)):
        with open(paths[i], "rb") as file:
            prediction_dict = pickle.load(file)
        with open(os.path.join(config.DATASET_DIRECTORY, prediction_dict["testing_dataset"], config.WRAPPER_DICT_NAME), "rb") as file:
            dataset = pickle.load(file)["dataset"]

        pt[i] = prediction_dict["predicted_pt"]
        event_correspondance[i] = dataset.event_correspondance

        if i == 0:
            nevents = dataset.events_processed
        elif dataset.events_processed != nevents:
            raise ValueError("All datasets must have the same number of events")
    
    pt = np.concatenate(pt)
    event_correspondance = np.concatenate(event_correspondance)

    leading_muon_pt = lm_pt(pt, event_correspondance)
    return leading_muon_pt, nevents

def get_rate(leading_muon_pt, nevents, pt_cut):
    return (np.sum(leading_muon_pt > pt_cut) / nevents) * COLLISION_FREQ

def rate_plot(leading_muon_pt, nevents, pt_cuts=np.linspace(0, 50, 51)):
    fig, ax = plt.subplots(1)
    
    passing_muons = np.zeros(pt_cuts)
    for i in range(len(pt_cuts)):
        passing_muons[i] = np.sum(leading_muon_pt > pt_cuts[i])
    
    rates = (passing_muons / nevents) * COLLISION_FREQ / 1000
    ax.scatter(pt_cuts, rates)

    ax.set_xlabel(r"$p_T$ threshold (GeV)")
    ax.set_ylabel("Rate (kHz)")
    ax.set_yscale("log")

    return fig, ax