from Dataset.constants import Run3TrainingVariables
from Model.BDT import BDT
import numpy as np

def current_EMTF(mode):
    use_features = Run3TrainingVariables[str(mode)]
    return current_BDT_anyfeatures(use_features)

def pT_to_target_log(x):
    return np.nan_to_num(np.log(x))
def target_to_pT_log(x):
    return np.nan_to_num(np.exp(x))
def pt_weighting_1overlog2(x):
    return np.nan_to_num(1 / np.log2(x))

def current_BDT_anyfeatures(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log,
        target_to_pT    = target_to_pT_log,
        pT_weighting    = pt_weighting_1overlog2,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model


# We are having trouble reproducing the previous BDT results. Lets try different event weights and 

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_1overlog2(x):
    return np.nan_to_num(1 / np.log2(x))

def model_target_log2_weighting_1overlog2(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overlog2,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model


def model_esitmators_800_depth_5(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overlog2,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 800
    )
    return model

def model_esitmators_400_depth_10(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overlog2,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 10,
        n_estimators    = 400
    )
    return model

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_1overlog100(x):
    return np.nan_to_num(1 / np.emath.logn(100, x))

def model_target_log2_weighting_1overlog100(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overlog100,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_uniform(x):
    return np.ones(len(x))

def model_target_log2_weighting_uniform(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_uniform,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model


def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_lessthan100(x):
    return np.array(x < 100, dtype=float)

def model_target_log2_weighting_lessthan100(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_lessthan100,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_1overpT(x):
    return 1/x

def model_target_log2_weighting_1overpT(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overpT,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_1overpTsquared(x):
    return 1/(x ** 2)

def model_target_log2_weighting_1overpTsquared(use_features):
    model = BDT(
        features        = use_features,
        pT_to_target    = pT_to_target_log2,
        target_to_pT    = target_to_pT_log2,
        pT_weighting    = pt_weighting_1overpTsquared,
        objective       = "reg:squarederror",
        learning_rate   = 0.1,
        max_depth       = 5,
        n_estimators    = 400
    )
    return model