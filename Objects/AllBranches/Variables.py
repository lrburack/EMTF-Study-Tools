from Dataset import TrainingVariable
from constants import *

class TrackVariables(TrainingVariable):
    def __init__(self, branches):
        self.branches = branches
        super().__init__(branches, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for i, branch in enumerate(self.branches):
            self.feature_inds[i] = getattr(event["EMTFNtuple"], branch)[int(self.shared_reference.track)]

class HitVariables(TrainingVariable):
    def __init__(self, branches, stations):
        feature_names = []
        for branch in branches:
            for station in stations:
                feature_names += [branch + "_" + str(station + 1)]
        self.branches = branches
        self.stations = stations
        super().__init__(feature_names, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for i, branch in enumerate(self.branches):
            for j, station in enumerate(self.stations):
                ind = i * len(self.stations) + j
                self.feature_inds[ind] = getattr(event["EMTFNtuple"], branch)[int(self.shared_reference.hitrefs[station])]