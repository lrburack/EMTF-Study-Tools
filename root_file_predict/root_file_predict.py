import pickle
import numpy as np
import ROOT
import sys

sys.path.append("/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools")

from Dataset.Dataset import *
from Dataset.Default.Variables import *
from Dataset.Default.SharedInfo import *
from Dataset.Default.TrackSelectors import *
from Dataset.AllBranches.Variables import *

# Script thats easy to run that takes a root file and paths to trained bdts for each mode. 
# Predicts pt for each track using the correct BDT and puts the result in the root file. 
# Forgive the ugliness of this script -- this codebase was not designed to be used this way.

# PREP EVERYTHING WE NEED: 
# 1. A root file with tracks to predict pt for.
# 2. Paths to trained BDTs for each mode.
# 3. A Dataset object for each mode to process tracks for BDT input.

# Only works with one file at a time for now
base_dirs = [sys.argv[1]]
out_path = sys.argv[2]

event_data, file_names = Dataset.get_root(base_dirs, files_per_endcap=1)

# Trained BDTs for each mode to predict pt
bdt_paths = {
    mode: f"/eos/home-l/lburack/work/BDT_studies/Results/Studies/BDT2025/NewTraining/mode={mode}/mode={mode}_model.pkl"
    for mode in [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
}

# Dataset object to process each track and compute BDT inputs. 
# Make sure you include the Track() variable in the variables list for each mode. -- this is needed for adding predictions to the root file
datasets = {
    mode: Dataset(variables=[
        Theta.for_mode(mode),
        St1_Ring2.for_mode(mode),
        dPhi.for_mode(mode),
        dTh.for_mode(mode),
        FR.for_mode(mode),
        RPC.for_mode(mode),
        Bend.for_mode(mode),
        Track.for_mode(mode),
    ], track_selector=TrackSelector(mode=mode, include_mode_15=False), # Make sure include_mode_15 is false otherwise this script will take so so long to run 
    shared_info=SharedInfo(mode=mode)
    )
    for mode in bdt_paths.keys()
}

datasets[15] = Dataset(variables=[
    # Add the necessary variables for mode 15
    Theta.for_mode(15),
    St1_Ring2.for_mode(15),
    dPhi.for_mode(15),
    dTh.for_mode(15),
    FR.for_mode(15),
    RPC.for_mode(15),
    Bend.for_mode(15),
    OutStPhi.for_mode(15),
    dPhiSum4.for_mode(15),
    dPhiSum4A.for_mode(15),
    dPhiSum3.for_mode(15),
    dPhiSum3A.for_mode(15),
    Track.for_mode(15),
], track_selector=TrackSelector(mode=15), shared_info=SharedInfo(mode=15))  # Adjust as needed

# ------------ EDIT ABOVE HERE ----------------

# Load the BDTs from the specified paths
bdts = {
    mode: pickle.load(open(bdt_paths[mode], "rb"))
    for mode in bdt_paths.keys()
}

print(event_data)

event_count = event_data[list(event_data.keys())[0]].GetEntries()
# preallocate the space for the predicted pts. Each row correponds to an event, each column to a track.
# EMTF has a maximum of 24 tracks per event, so we can preallocate a 24 column array
predicted_pts = np.full((event_count, 24), np.nan, dtype=np.float32)

print(event_count)

# Compute BDT inputs for each mode and predict pt
for mode in bdt_paths.keys():
    print(f"root_file_predict: ---------------------------\nPROCESSING MODE {mode}\n---------------------------")
    bdt = bdts[mode]
    dataset = datasets[mode]
    
    print("root_file_predict: Building dataset")
    dataset.build_dataset(event_data)
    predicted_pt = bdt.predict(bdt.prep_events(dataset.data, dataset.feature_names))

    predicted_pts[dataset.event_correspondance, dataset.get_features("track").astype("int")] = predicted_pt

print(f"root_file_predict: ---------------------------\nWRITING TO ROOT FILE\n---------------------------")

# My gratitude to ChatGPT for everything below :)

import ROOT
import array

from ctypes import pointer
from array import array
from ROOT import std

import time

start = time.time()

# Open the original ROOT file (read-only)
input_file = ROOT.TFile.Open(file_names[0], "READ")
input_tree = input_file.Get("EMTFNtuple/tree")  # Get tree from subdirectory

# Create a new ROOT file in the desired location
output_file_path = os.path.join(out_path, "modified_file.root")
output_file = ROOT.TFile.Open(output_file_path, "RECREATE")

# Clone the tree structure but without copying entries yet
new_tree = input_tree.CloneTree(0)

# Create a std::vector<float> branch to hold predicted pts per event
pt_vec = std.vector('float')()
pt_branch = new_tree.Branch("new_predicted_pt", pt_vec)

# Loop over events in original tree, fill new tree with new branch
for i_event in range(input_tree.GetEntries()):
    input_tree.GetEntry(i_event)
    
    pt_vec.clear()
    
    track_pts = predicted_pts[i_event]  # Replace with your predicted pts source
    n_tracks_to_write = getattr(input_tree, 'emtfTrack_size')  # Number of tracks in this event

    for i in range(n_tracks_to_write):
        if track_pts[i] != np.nan:
            pt_vec.push_back(track_pts[i])
        else:
            pt_vec.push_back(0)
    
    new_tree.Fill()

# Write new tree and close files
output_file.cd()
new_tree.Write()
output_file.Close()
input_file.Close()

print(time.time() - start)
