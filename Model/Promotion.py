from Model.Model import Model
import numpy as np

class Promotion(Model):
    def __init__(self, model: Model, promotion_features, promotion_function):
        self.model = model
        self.features = promotion_features
        self.promotion_function = promotion_function
    
    def train(self, events, target):
        self.model.train(events[:, :len(self.model.features)], target)

    def predict(self, events):
        pT = self.model.predict(events[:, :len(self.model.features)])

        return self.promotion_function(pT, events[:, -len(self.features):])
    
    def prep_events(self, events, feature_names):
        return np.column_stack((self.model.prep_events(events, feature_names), super().prep_events(events, feature_names)))