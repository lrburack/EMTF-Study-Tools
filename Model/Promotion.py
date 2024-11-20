import Model.Model

class Promotion(Model):
    def __init__(self, model: Model, promotion_features, promotion_function):
        self.model = model
        self.features = promotion_features
        self.promotion_function = promotion_function
    
    def train(self, events, target):
        train_events = self.model.prep_events(events)
        self.model.train(train_events, target)

    def predict(self, events):
        test_events = self.model.prep_events(events)
        pT = self.model.predict(test_events)

        promotion_info = self.prep_events(events)

        return self.promotion_function(pT, promotion_info)