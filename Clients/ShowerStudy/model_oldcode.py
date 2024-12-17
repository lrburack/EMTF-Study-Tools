# ML modules
from Model.Implementations.Default import *
from Model.Implementations.PreviousBDT import *

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, get_combined_dataset
import sys

mode = int(sys.argv[1])

# The study directory to output the results to
study = "ReproducePreviousBDT/OldCode/Rates"
# study = "ReproducePreviousBDT/wHMT"
name = f"mode={mode}"
os.makedirs(os.path.join(config.STUDY_DIRECTORY, study), exist_ok=True)


# -------------------------------- GETTING A MODEL -------------------------------
# Two options (comment out one of them completely):
#   OPTION 1. Train a new model --------------------

#   OPTION 2. Load an existing model ---------------

xgb_model_path = f"/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools/Validation/Results/mode={mode}/xgb_model.pkl"

with open(xgb_model_path, "rb") as file:
    xgb_model = pickle.load(file)

model = model_target_log2_weighting_1overlog2(Run3TrainingVariables[str(mode)])
model.bdt = xgb_model
model.trained = True

# -------------------------------- TESTING ---------------------------------------
# The dataset on which to test the model
# testing_dataset_names = [f"Control/Compressed/mode={mode}_testing_distribution"]
testing_dataset_names = [f"ReproducePreviousBDT/Rates/mode={mode}"]
# testing_dataset_names = [f"ReproducePreviousBDT/wHMT_testing_distribution/mode={mode}"]
testing_dataset = get_combined_dataset(testing_dataset_names)
# The indices of the events in the testing dataset to use. Leave None for all events
tracks_to_test = None

# -------------------------------- EDIT ABOVE THIS -------------------------------



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