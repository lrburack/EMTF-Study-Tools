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
study = "ShowerStudy/AllStats/Uncompressed/AllShowerInfo/"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)

base_dataset = f"Control/AllStats/Uncompressed/mode={mode}"

# -------------------------------- GETTING A MODEL -------------------------------
# Two options (comment out one of them completely):
#   OPTION 1. Train a new model --------------------
training_dataset_names = [base_dataset, f"ShowerStudy/AllStats/mode={mode}"]
training_dataset = get_combined_dataset(training_dataset_names)

print(training_dataset.feature_names)

# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_train = None
# To design your own model, implement it in Model/Implementations and import it
# model = current_EMTF(mode=mode)
shower_features = ['loose_1', 'nominal_1', 'tight_1', 'loose_2', 'nominal_2', 'tight_2', 'loose_3', 'nominal_3', 'tight_3', 'loose_4', 'nominal_4', 'tight_4', 'loose_showerCount', 'nominal_showerCount', 'tight_showerCount', 'careful_shower_bit_thresh=1', 'careful_shower_bit_thresh=2', 'careful_shower_bit_thresh=3', 'shower_type_0', 'shower_type_1', 'shower_type_2', 'shower_type_3']
use_features = Run3TrainingVariables[str(mode)] + list(np.array(shower_features)[np.isin(np.array(shower_features), training_dataset.feature_names)])
# use_features = Run3TrainingVariables[str(mode)] + ["careful_shower_bit_thresh=1"]
model = current_BDT_anyfeatures(use_features)
print(model.features)

#   OPTION 2. Load an existing model ---------------
# model = f"Control/Uncompressed/mode={mode}_{config.MODEL_NAME}"

# -------------------------------- TESTING ---------------------------------------
# The dataset on which to test the model
testing_dataset_names = [f"Control/Uncompressed/mode={mode}_testing_distribution", f"ShowerStudy/mode={mode}_testing_distribution"]
testing_dataset = get_combined_dataset(testing_dataset_names)
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_test = None

# -------------------------------- EDIT ABOVE THIS -------------------------------


if tracks_to_train == None:
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
if tracks_to_test == None:
    tracks_to_test = np.arange(testing_dataset.tracks_processed)

testing_events = model.prep_events(testing_dataset.data, testing_dataset.feature_names)[tracks_to_test]
print(f"* Testing model on {len(testing_events)} tracks")
predicted_pt = model.predict(testing_events)

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