from Dataset.Dataset import TrainingVariable
from Dataset.constants import *

from Dataset.Default.Variables import GeneratorVariables

class RecoVariables(GeneratorVariables):
    def __init__(self):
        super().__init__()
    
    def calculate(self, event):
        self.feature_inds[0] = event["EMTFNtuple"].recoMuon_pt[self.shared_reference.reco_match]
        self.feature_inds[1] = event["EMTFNtuple"].recoMuon_eta[self.shared_reference.reco_match]
        self.feature_inds[2] = event["EMTFNtuple"].recoMuon_phi[self.shared_reference.reco_match]