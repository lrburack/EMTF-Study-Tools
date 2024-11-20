import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from Model.Model import Model
import numpy as np

class BDT(Model):
    def __init__(self, features, pT_to_target, target_to_pT, pT_weighting, objective, learning_rate, max_depth, n_estimators):
        super().__init__(features, pT_to_target, target_to_pT, pT_weighting)
        
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.bdt = xgb.XGBRegressor(objective = objective, 
                        learning_rate = self.learning_rate, 
                        max_depth = self.max_depth, 
                        n_estimators = self.n_estimators,
                        nthread = 30)
    
    def train(self, events, true_pT):
        events = self.check_valid(events)
        self.trained = True

        target = self.pT_to_target(true_pT)
        weights = self.pT_weighting(true_pT)
        self.bdt.fit(events, target, sample_weight = weights)

    def predict(self, events):
        events = self.check_valid(events)
        if not self.trained:
            raise RuntimeError("The model cannot predict before it has been trained")
        
        return self.target_to_pT(self.bdt.predict(events))
        
