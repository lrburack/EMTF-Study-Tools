# ML modules
from Model.Implementations.Default import *

import pickle
import config
import os
import numpy as np
from helpers import get_by_name
import sys

mode = int(sys.argv[1])

# To make the control dataset just re

# The study directory to output the results to
study = "NewBend/"
# study = "Control/"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)

# -------------------------------- GETTING A MODEL -------------------------------
# To design your own model, implement it in Model/Implementations and import it
# For newbend
model = current_BDT_anyfeatures(Run3TrainingVariables[str(mode)] + ["GEMCSC_dPhi", "GEM_layer"])
# For control
# model = current_BDT_anyfeatures(Run3TrainingVariables[str(mode)])

dataset_path = f"NewBend/mode={mode}"

with open(os.path.join(config.DATASET_DIRECTORY, dataset_path, config.WRAPPER_DICT_NAME), "rb") as file:
    dataset = pickle.load(file)["dataset"]

dataset.randomize_event_order()

# train on the first half
tracks_to_train = np.arange(dataset.tracks_processed // 2)

# test on the second half
tracks_to_test = np.arange(dataset.tracks_processed // 2, dataset.tracks_processed)

# -------------------------------- EDIT ABOVE THIS -------------------------------


# Get the model. Check which option was selected
training_dataset = dataset
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
testing_dataset = dataset
if tracks_to_test is None:
    tracks_to_test = np.arange(testing_dataset.tracks_processed)

testing_events = model.prep_events(testing_dataset.data, testing_dataset.feature_names)[tracks_to_test]
print(f"* Testing model on {len(testing_events)} tracks")
predicted_pt = model.predict(testing_events)

test_dict = {
    "model_path"        : model_name,
    "testing_dataset"   : testing_dataset,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)