from Dataset.constants import Run3TrainingVariables
from Model.BDT import BDT
from Model.Promotion import Promotion
import numpy as np

def pT_to_target(x):
    return np.nan_to_num(np.log(x))
def target_to_pT(x):
    return np.nan_to_num(np.exp(x))
def pt_weighting(x):
    return np.nan_to_num(1 / np.log2(x))

# def all_shower_info(mode):
#     use_features = Run3TrainingVariables[str(mode)]
#     model = BDT(
#         features        = use_features,
#         pT_to_target    = pT_to_target,
#         target_to_pT    = target_to_pT,
#         pT_weighting    = pt_weighting,
#         objective       = "reg:squarederror",
#         learning_rate   = 0.1,
#         max_depth       = 5,
#         n_estimators    = 400
#     )
#     return model

def pT_to_target_log2(x):
    return np.nan_to_num(np.log2(x))
def target_to_pT_log2(x):
    return np.nan_to_num(np.power(2, x))
def pt_weighting_1overlog2(x):
    return np.nan_to_num(1 / np.log2(x))

def promotion_function_1_loose(pT, promotion_features):
    pT[promotion_features[:, 0] >= 1] = 1000
    return pT

def promote_1_loose(mode):
    use_features = Run3TrainingVariables[str(mode)]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount"],
        promotion_function=promotion_function_1_loose
    )
    return model

def promotion_function_2_loose(pT, promotion_features):
    pT[promotion_features[:, 0] >= 2] = 1000
    return pT

def promote_2_loose(mode):
    use_features = Run3TrainingVariables[str(mode)]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount"],
        promotion_function=promotion_function_2_loose
    )
    return model

def promotion_function_1_nominal(pT, promotion_features):
    pT[promotion_features[:, 0] >= 1] = 1000
    return pT

def promote_1_nominal(mode):
    use_features = Run3TrainingVariables[str(mode)]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["nominal_showerCount"],
        promotion_function=promotion_function_1_nominal
    )
    return model


def promotion_function_2_loose_or_1_nominal(pT, promotion_features):
    pT[(promotion_features[:, 0] >= 2) | (promotion_features[:, 1] >= 1)] = 1000
    return pT

def promote_2_loose_or_1_nominal(mode):
    use_features = Run3TrainingVariables[str(mode)]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount","nominal_showerCount"],
        promotion_function=promotion_function_2_loose_or_1_nominal
    )
    return model


def promote_2_loose_or_1_nominal_and_looseshowercount(mode):
    use_features = Run3TrainingVariables[str(mode)] + ["loose_showerCount"]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount","nominal_showerCount"],
        promotion_function=promotion_function_2_loose_or_1_nominal
    )
    return model

def promote_2_loose_or_1_nominal_and_looseshowerbit(mode):
    use_features = Run3TrainingVariables[str(mode)] + ["careful_shower_bit_thresh=1"]
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount","nominal_showerCount"],
        promotion_function=promotion_function_2_loose_or_1_nominal
    )
    return model

def promote_2_loose_or_1_nominal_anyfeatures(use_features):
    model = Promotion(
        model = BDT(
            features        = use_features,
            pT_to_target    = pT_to_target_log2,
            target_to_pT    = target_to_pT_log2,
            pT_weighting    = pt_weighting_1overlog2,
            objective       = "reg:squarederror",
            learning_rate   = 0.1,
            max_depth       = 5,
            n_estimators    = 400
        ),
        promotion_features=["loose_showerCount","nominal_showerCount"],
        promotion_function=promotion_function_2_loose_or_1_nominal
    )
    return model