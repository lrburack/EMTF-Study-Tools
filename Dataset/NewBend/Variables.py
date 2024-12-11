from Dataset.Dataset import TrainingVariable
from Dataset.constants import *

# event is a dict with a key for every tree in the Ntuple
# MuShowerNtuple
# EMTFNtuple

class NewBend(TrainingVariable):
    def __init__(self):
        super().__init__(["GEMCSC_dPhi", "GEM_layer"], tree_sources=["EMTFNtuple"])
    
    def calculate(self, event):

        GEM_hitrefs = np.where(np.array(event["EMTFNtuple"].emtfHit_type) == 3)
        phi = np.abs(event["EMTFNtuple"].emtfTrack_phi[self.shared_reference.track]) * (np.pi / 180)
        eta = event["EMTFNtuple"].emtfTrack_eta[self.shared_reference.track]

        dR = np.zeros(GEM_hitrefs)

        for i, GEM_hitref in enumerate(GEM_hitrefs):
            # For each GEM hit you have to calculate the dR between the gem hit and the station 1 csc
            dR[i] = 

        # To get the track, self.shared_reference.track <- just an int
        # To get the hitrefs, self.shared_reference.hitrefs <- a np array of length 4
        event["EMTFNtuple"].emtfTrack_ptLUT_signPh[self.shared_reference.track][0]

        self.feature_inds[0] = 0 # Put GEMCSC_dPhi here
        self.feature_inds[1] = 0 # GEM_layer
        # dont need to return anything