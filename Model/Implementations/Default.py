from Dataset.constants import Run3TrainingVariables
from Model.BDT import BDT
import numpy as np

def current_EMTF(mode):
    use_features = Run3TrainingVariables[str(mode)]
    return current_BDT_anyfeatures(use_features)

def pT_to_target(x):
    return np.nan_to_num(np.log(x))
def target_to_pT(x):
    return np.nan_to_num(np.exp(x))
def pt_weighting(x):
    return np.nan_to_num(1 / np.log2(x))

def current_BDT_anyfeatures(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target,
        target_to_pT    = target_to_pT,
        pT_weighting    = pt_weighting,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model