from Dataset.Dataset import SharedInfo
from Dataset.constants import *

class RecoSharedInfo(SharedInfo):
    def __init__(self, mode):
        self.reco_match = None
        super().__init__(mode)
    
    # This could be further optimized, but its already plenty fast. It is silly to repeat much of the same dR math done in the track selector, but thats okay 
    def calculate(self, event, track):
        super().calculate(event, track)

        # We use the reco muon as the gen muon, but we must figure out which track in the EMTF corresponds to which gen muon
        # To do this we calculate dR=sqrt(dPhi^2 + dEta^2) using station 2 phi and eta, and find the minimum. 
        phi = np.abs(event["EMTFNtuple"].emtfUnpTrack_phi[self.track]) * (np.pi / 180)
        eta = event["EMTFNtuple"].emtfUnpTrack_eta[self.track]
        phi_diffs = np.abs(np.array(event["EMTFNtuple"].recoMuon_phiSt2)) - phi
        eta_diffs = np.array(event["EMTFNtuple"].recoMuon_etaSt2) - eta

        dR = np.sqrt(eta_diffs ** 2 + phi_diffs ** 2)
        
        self.reco_match = int(np.argmin(dR))