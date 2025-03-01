# ML modules
from Model.Implementations.Default import *

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, permute_together
import sys

mode = 15

# The study directory to output the results to
study = "Tutorial/"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)

# -------------------------------- GETTING A MODEL -------------------------------
# Two options (comment out one of them completely):
#   OPTION 1. Train a new model --------------------
training_dataset_name = f"Tutorial/mode={mode}"
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_train = None
# To design your own model, implement it in Model/Implementations and import it
model = current_EMTF(mode=mode)

#   OPTION 2. Load an existing model ---------------
# model = f"Control/Uncompressed/mode={mode}_{config.MODEL_NAME}"

# -------------------------------- TESTING ---------------------------------------
# The dataset on which to test the model
testing_dataset_name = f"Tutorial/mode={mode}_testing_distribution"
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_test = None

# -------------------------------- EDIT ABOVE THIS -------------------------------

# Get the model. Check which option was selected
if isinstance(model, str):  # option 2
    with open(os.path.join(config.STUDY_DIRECTORY, model), "rb") as file:
        model = pickle.load(file)
    use_features = model.features
else:                       # option 1
    training_dataset = get_by_name(training_dataset_name)["dataset"]
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
testing_dataset = get_by_name(testing_dataset_name)["dataset"]
if tracks_to_test is None:
    tracks_to_test = np.arange(testing_dataset.tracks_processed)

testing_events = model.prep_events(testing_dataset.data, testing_dataset.feature_names)[tracks_to_test]
print(f"* Testing model on {len(testing_events)} tracks")
predicted_pt = model.predict(testing_events)

test_dict = {
    "model_path"        : model_name,
    "testing_dataset"   : testing_dataset_name,
    "testing_tracks"    : tracks_to_test,
    "predicted_pt"      : predicted_pt,
}

# Save the prediction
prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)
