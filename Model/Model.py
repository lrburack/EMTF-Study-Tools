from typing import List
import numpy as np

# A generic superclass for models
# We want to be able to train and test many different types of models with generic methods
class Model:
    def __init__(self, features: List[str], pT_to_target, target_to_pT, pT_weighting):
        self.pT_to_target = pT_to_target
        self.target_to_pT = target_to_pT
        self.pT_weighting = pT_weighting

        self.features = features
        self.trained = False

    def train(self, events, true_pT):
        events = self.check_valid(events)
        self.trained = True

    def predict(self, events):
        events = self.check_valid(events)
        if not self.trained:
            raise RuntimeError("The model cannot predict before it has been trained")
        return np.zeros(len(events))

    def check_valid(self, events):
        shape = np.shape(events)
        if len(shape) == 2:
            if shape[1] != len(self.features):
                raise ValueError(f"The number of features passed ({shape[1]}) does not match the number accepted by this model ({len(self.features)})")
            return events
        elif len(shape) == 1: # a single event was passed
            if shape[0] != len(self.features):
                raise ValueError(f"The number of features passed ({shape[0]}) does not match the number accepted by this model ({len(self.features)})")
            return np.array([events])
        else:
            raise ValueError("Invalid shape for events array")
    
    def prep_events(self, events, feature_names):
        indices = np.zeros(len(self.features), dtype=np.int_)
        for i, feature in enumerate(self.features):
            if feature not in feature_names:
                raise ValueError(f"The passed events do not contain the required feature {feature}")
            indices[i] = feature_names.index(feature)

        return events[:, indices]