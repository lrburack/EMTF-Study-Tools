# ML modules
from Model.Implementations.Default import *

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, permute_together
import sys

mode = int(sys.argv[1])

# The study directory to output the results to
study = "Tests/Condor/FullFlow"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)

# ---------------------- GETTING A MODEL ------------------
# Two options (comment out one of them completely):
#   1. Load an existing model -----------------------------
# model_name = "some/path"
# with open(os.path.join(config.STUDY_DIRECTORY, study, model_name), "wb") as file:
#     model = pickle.load(file)

# use_features = model.features

#   2. Train a new model ----------------------------------
training_dataset_name = f"Tests/Condor/FullFlow/mode={mode}"
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_train = None

training_dataset = get_by_name(training_dataset_name)["dataset"]
if tracks_to_train == None:
    tracks_to_train = np.arange(training_dataset.tracks_processed)

# Create the model. Its best to create a function in Model/Implementations to create your model.
model = current_EMTF(mode=mode)
# use_features = np.array(training_dataset.feature_names)[training_dataset.trainable_features] # All features
# use_features = Run3TrainingVariables[str(mode)] + ["my", "other", "features"]      # Some features
# model = current_BDT_anyfeatures(use_features)

training_events = model.prep_events(training_dataset.data, training_dataset.feature_names)[tracks_to_train]
training_pt = training_dataset.get_features("gen_pt")[tracks_to_train]

print(f"* Training model on {len(training_events)} tracks")
model.train(training_events, training_pt)

model_name = f"{name}_{config.MODEL_NAME}"
with open(os.path.join(config.STUDY_DIRECTORY, study, model_name), "wb") as file:
    pickle.dump(model, file)

# ---------------------- TESTING --------------------------
# The dataset on which to test the model
testing_dataset_name = f"Tests/Condor/FullFlow/mode={mode}_testing_distribution"
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_test = None

testing_dataset = get_by_name(testing_dataset_name)["dataset"]
if tracks_to_test == None:
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

prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
print(prediction_path)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)