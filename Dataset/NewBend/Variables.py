from Dataset.Dataset import TrainingVariable
from Dataset.constants import *

from Dataset.NewBend.NewBend_LUT import CSCGEMSlopecorrector

# event is a dict with a key for every tree in the Ntuple
# MuShowerNtuple
# EMTFNtuple

class NewBend(TrainingVariable):
    def __init__(self, dR_match_max: float = None):
        self.dR_match_max = dR_match_max
        super().__init__(["GEMCSC_dPhi", "GEM_layer"], tree_sources=["EMTFNtuple"])
    
    def calculate(self, event):
        GEM_hitrefs = np.where(np.array(event["EMTFNtuple"].emtfHit_type) == 3)[0]

        me1phi = event["EMTFNtuple"].emtfHit_emtf_phi[int(self.shared_reference.hitrefs[0])]
        me1theta = event["EMTFNtuple"].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[0])]
        slope = event["EMTFNtuple"].emtfHit_slope[int(self.shared_reference.hitrefs[0])]
        slope_sign = 1 if event["EMTFNtuple"].emtfHit_bend[int(self.shared_reference.hitrefs[0])] % 2 == 0 else -1                 # Odd bend means negative slope
        isME1a = event["EMTFNtuple"].emtfHit_ring[int(self.shared_reference.hitrefs[0])] == 4    # Ring 4 means ME1a

        # We will look for the closest GEM hit to the station 1 hit in the track, and then use that
        # for the GEM_CSC_dphi and GEM_layer
        closest_dR = np.inf
        GEMCSC_dphi = -99
        GEM_layer = -99

        # We need to find the closest GEM hit
        for GEM_hitref in GEM_hitrefs:
            GEM_hitref = int(GEM_hitref)

            dtheta = event["EMTFNtuple"].emtfHit_emtf_theta[GEM_hitref] - me1theta
            dphi = event["EMTFNtuple"].emtfHit_emtf_phi[GEM_hitref] - me1phi

            # For calculating dR we will use a corrected dphi value
            slope_corr = CSCGEMSlopecorrector(
                slope, 
                event["EMTFNtuple"].emtfHit_layer[GEM_hitref], 
                event["EMTFNtuple"].emtfHit_chamber[GEM_hitref],
                slope_sign,
                isME1a
            )
            dphi_corr = dphi - slope_corr

            # For each GEM hit you have to calculate the dR between the gem hit and the station 1 csc
            dR = np.sqrt(dphi_corr**2 + dtheta ** 2)

            if dR < closest_dR:
                closest_dR = dR
                GEMCSC_dphi = dphi # We use the corrected dphi only for matching the gem hit. We store the uncorrected dphi
                GEM_layer = event["EMTFNtuple"].emtfHit_layer[GEM_hitref]
        

        # If a max dR cut was specified we will return -99 values. In principle, if the NewBendTrackSelector is used, this should never happen
        if self.dR_match_max is not None and closest_dR > self.dR_match_max:
            self.feature_inds[0] = -99
            self.feature_inds[1] = -99
            return

        self.feature_inds[0] = GEMCSC_dphi # Put GEMCSC_dPhi here
        self.feature_inds[1] = GEM_layer # GEM_layer
        # dont need to return anything