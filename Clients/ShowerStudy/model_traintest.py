# ML modules
from Model.Implementations.Default import *
from Model.Implementations.PreviousBDT import *
from Model.Implementations.ShowerStudy import *
from Dataset.Dataset import Dataset

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, get_combined_dataset
import sys

mode = int(sys.argv[1])

# The study directory to output the results to
# study = "ReproducePreviousBDT/BDT2023_noGEM"
# study = "ReproducePreviousBDT/AllStats/LikePrevious"
# study = "DebugSpike/NoGMTConversion"
# study = "BDT2025/ShowerStudy/LooseShowerCount/SectorMatching/"
# study = "BDT2025/ShowerStudy/Promotion/SectorMatching/1Nominal"
# study = "BDT2025/ShowerStudy/Promotion/SectorMatching/2Loose_or_1Nominal"
# study = "BDT2025/ShowerStudy/Promotion/SectorMatching/2Loose_or_1Nominal_and_LooseShowerCount"
# study = f"BDT2025/TrueEfficiency/mode={mode}"
# study = "BDT2025/ShowerStudy/Realistic/SectorMatching/remove_outer_FR_add_LooseShowerBit_To3StationOnly_promote_2L_or_1N"
# study = "BDT2025/NewControl"
study = "BDT2025/Control/"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study, "Rates/"), exist_ok=True)
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study, "TrackPt/"), exist_ok=True)
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study, "TrackPt/Rates"), exist_ok=True)



# -------------------------------- GETTING A MODEL -------------------------------
training_dataset = get_combined_dataset([f"BDT2025/Control/mode={mode}"])#, f"BDT2025/ShowerStudy/MatchChamber/mode={mode}"])
training_dataset.randomize_event_order()

print(f"The training data has {training_dataset.tracks_processed} total tracks")
count = np.sum(training_dataset.get_features("emtfTrack_mode") != mode)
print(f"Forced mode tracks: {count}")
count = np.sum(training_dataset.get_features("emtfTrack_mode") == mode)
print(f"Mode {mode} tracks: {count}")

print(training_dataset.feature_names)

# The indices of the events in the testing dataset to use. Leave None for all events
# tracks_to_train = np.arange(training_dataset.tracks_processed // 2)
tracks_to_train = None

# shower_features = ['loose_1', 'nominal_1', 'tight_1', 'loose_2', 'nominal_2', 'tight_2', 'loose_3', 'nominal_3', 'tight_3', 'loose_4', 'nominal_4', 'tight_4', 'loose_showerCount', 'nominal_showerCount', 'tight_showerCount', 'careful_shower_bit_thresh=1', 'careful_shower_bit_thresh=2', 'careful_shower_bit_thresh=3', 'shower_type_0', 'shower_type_1', 'shower_type_2', 'shower_type_3']
# available_shower_features = [feature for feature in shower_features if feature in training_dataset.feature_names]
# available_shower_features = ["careful_shower_bit_thresh=1"]
# available_shower_features = []
# exclude_features = ["FR_2", "FR_3", "FR_4"]
# use_features = [feature for feature in Run3TrainingVariables[str(mode)] if feature not in exclude_features] + available_shower_features

# To design your own model, implement it in Model/Implementations and import it
model = model_target_log2_weighting_1overlog2(Run3TrainingVariables[str(mode)])
# model = model_target_log2_weighting_1overlog2(use_features)
# model = promote_2_loose_or_1_nominal(mode)
# model = promote_2_loose_or_1_nominal_and_looseshowercount(mode)
# model = promote_2_loose_or_1_nominal_anyfeatures(use_features)
# print(model.features)

# -------------------------------- TESTING ---------------------------------------
# The indices of the events in the testing dataset to use. Leave None for all events

# testing_dataset = get_combined_dataset([f"BDT2025/Control/mode={mode}_testing_distribution"])#, f"BDT2025/ShowerStudy/MatchChamber/mode={mode}_testing_distribution"])
testing_dataset = get_combined_dataset([f"BDT2025/Control/ForcedModeTestingDistributions/mode={mode}"])
tracks_to_test = None
# print(np.any(np.isin(tracks_to_train, tracks_to_test)))


# -------------------------------- EDIT ABOVE THIS -------------------------------


if tracks_to_train is None:
    tracks_to_train = np.arange(training_dataset.tracks_processed)
training_events = model.prep_events(training_dataset.data, training_dataset.feature_names)[tracks_to_train]

training_pt = training_dataset.get_features("gen_pt")[tracks_to_train]

print(f"* Training model on {len(training_events)} tracks")
model.train(training_events, training_pt)

# Save the trained model
model_name = f"{name}_{config.MODEL_NAME}"
with open(os.path.join(config.STUDY_DIRECTORY, study, model_name), "wb") as file:
    pickle.dump(model, file)

# Testing
if tracks_to_test is None:
    tracks_to_test = np.arange(testing_dataset.tracks_processed)

testing_events = model.prep_events(testing_dataset.data, testing_dataset.feature_names)[tracks_to_test]
print(f"* Testing model on {len(testing_events)} tracks")
predicted_pt = model.predict(testing_events)

# Transform to gmt pt
gmt_pt = np.array(((predicted_pt * 2) + 1), dtype=np.int_)
gmt_pt[gmt_pt > 511] = 511

# Transform back to pt
pt = (gmt_pt - 1) * 0.5
predicted_pt = pt
 
test_dict = {
    "model_path"        : os.path.join(config.STUDY_DIRECTORY, study, model_name),
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)

# Also get track pT

from Performance.scale_factor import current_EMTF_unscale_pt 
predicted_pt = testing_dataset.get_features(["emtfTrack_pt"])[tracks_to_test]
predicted_pt = current_EMTF_unscale_pt(predicted_pt) * predicted_pt

test_dict = {
    "model_path"        : "",
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, "TrackPt", name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)

# Also get rates

# The dataset on which to test the model
testing_dataset_names = [f"BDT2025/Control/Rates/mode={mode}"]#, f"BDT2025/ShowerStudy/MatchChamber/Rates/mode={mode}"]
testing_dataset = get_combined_dataset(testing_dataset_names)
tracks_to_test = np.arange(testing_dataset.tracks_processed)
# The indices of the events in the testing dataset to use. Leave None for all events

testing_events = model.prep_events(testing_dataset.data, testing_dataset.feature_names)[tracks_to_test]
print(f"* Testing model on {len(testing_events)} tracks")
predicted_pt = model.predict(testing_events)

# Transform to gmt pt
gmt_pt = np.array(((predicted_pt * 2) + 1), dtype=np.int_)
gmt_pt[gmt_pt > 511] = 511

# Transform back to pt
pt = (gmt_pt - 1) * 0.5
predicted_pt = pt

test_dict = {
    "model_path"        : os.path.join(config.STUDY_DIRECTORY, study, model_name),
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, "Rates", name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)

predicted_pt = testing_dataset.get_features("emtfTrack_pt")[tracks_to_test]
predicted_pt = current_EMTF_unscale_pt(predicted_pt) * predicted_pt

test_dict = {
    "model_path"        : os.path.join(config.STUDY_DIRECTORY, study, model_name),
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, "TrackPt/Rates", name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)