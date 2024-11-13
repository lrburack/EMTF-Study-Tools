from Dataset import TrackSelector
from constants import *

class RecoTrackSelector(TrackSelector):
    def __init__(self, mode: int, include_mode_15: bool = True, dR_match_max: float = 0.4):
        super().__init__(mode, include_mode_15)
        self.dR_match_max = dR_match_max

    def select(self, event: dict):
        good_tracks = super().select(event)

        reco_match = np.zeros(len(good_tracks), dtype=bool)

        # Loop through each track which was identified as 'good' and impose that it has a matching reco muon
        reco_phi = np.abs(np.array(event["EMTFNtuple"].recoMuon_phiSt2))
        reco_eta = np.array(event["EMTFNtuple"].recoMuon_etaSt2)
        for i, track in enumerate(good_tracks):
            phi = np.abs(event["EMTFNtuple"].emtfUnpTrack_phi[track]) * (np.pi / 180)
            eta = event["EMTFNtuple"].emtfUnpTrack_eta[track]
            dR = np.sqrt((eta - reco_eta) ** 2 + (phi - reco_phi) ** 2)
            if np.min(dR) < self.dR_match_max:
                reco_match[i] = True
        
        return good_tracks[reco_match]