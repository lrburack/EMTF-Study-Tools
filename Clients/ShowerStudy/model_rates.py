# ML modules
from Model.Implementations.Default import *

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, get_combined_dataset
import sys

mode = int(sys.argv[1])

# The study directory to output the results to
# study = "ShowerStudy/Rates/Uncompressed/AllShowerInfo/"
study = "BDT2025/Control/TrackPt/Rates"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)

base_dataset = f"ReproducePreviousBDT/Rates/mode={mode}"

# -------------------------------- GETTING A MODEL -------------------------------
# Two options (comment out one of them completely):
#   OPTION 1. Train a new model --------------------
# training_dataset_names = [base_dataset, f"ShowerStudy/mode={mode}"]
# training_dataset = get_combined_dataset(training_dataset_names)

# # The indices of the events in the testing dataset to use. Leave None for all events
# tracks_to_train = None
# # To design your own model, implement it in Model/Implementations and import it
# # model = current_EMTF(mode=mode)
# # shower_features = ['loose_1', 'nominal_1', 'tight_1', 'loose_2', 'nominal_2', 'tight_2', 'loose_3', 'nominal_3', 'tight_3', 'loose_4', 'nominal_4', 'tight_4', 'loose_showerCount', 'nominal_showerCount', 'tight_showerCount', 'careful_shower_bit_thresh=1', 'careful_shower_bit_thresh=2', 'careful_shower_bit_thresh=3', 'shower_type_0', 'shower_type_1', 'shower_type_2', 'shower_type_3']
# # use_features = Run3TrainingVariables[str(mode)] + list(np.array(shower_features)[np.isin(np.array(shower_features), training_dataset.feature_names)])
# use_features = Run3TrainingVariables[str(mode)] + ["careful_shower_bit_thresh=1"]
# model = current_BDT_anyfeatures(use_features)
# print(model.features)

#   OPTION 2. Load an existing model ---------------
# model = f"ReproducePreviousBDT/wHMT/mode={mode}_{config.MODEL_NAME}"
# model = f"ShowerStudy/Uncompressed/AllShowerInfo/mode={mode}_{config.MODEL_NAME}"

# -------------------------------- TESTING ---------------------------------------
# The dataset on which to test the model
# testing_dataset_names = [f"{base_dataset}", f"ShowerStudy/Rates/mode={mode}"]
testing_dataset_names = [f"{base_dataset}"]
testing_dataset = get_combined_dataset(testing_dataset_names)
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_test = None

# -------------------------------- EDIT ABOVE THIS -------------------------------

with open(os.path.join(config.STUDY_DIRECTORY, model), "rb") as file:
    model = pickle.load(file)

# Testing
if tracks_to_test == None:
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
    "model_path"        : model,
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)