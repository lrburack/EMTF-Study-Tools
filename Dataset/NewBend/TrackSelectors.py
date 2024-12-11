from Dataset.Dataset import TrackSelector
from Dataset.constants import *
from Dataset.NewBend import NewBend_LUT

class NewBendTrackSelector(TrackSelector):
    # This is made only for modes which contain station 1.
    def __init__(self, mode: int, include_mode_15: bool = True, dR_match_max: float = 0.15):
        if not get_station_presence(mode)[0]:
            raise ValueError("The NewBendTrackSelector can only be used for modes with station 1")
        
        super().__init__(mode, include_mode_15)
        self.dR_match_max = dR_match_max

    def select(self, event: dict):
        # We call the superclass TrackSelector which will just return a list of numbers
        # corresponding to tracks that have the right mode
        good_tracks = super().select(event)

        # Start by assuming that none of the tracks have a matching GEM hit
        has_GEM_match = np.zeros(len(good_tracks), dtype=bool)

        # Candidate 
        GEM_hitrefs = np.where(np.array(event["EMTFNtuple"].emtfHit_type) == 3)

        # Loop through each track which was identified as 'good' and check if it has a matching GEM hit
        for i, track in enumerate(good_tracks):
            # Remove station 1 ring 2 because this cant have a GEM match
            if event["EMTFNtuple"].emtfTrack_ptLUT_st1_ring2[track]:
                continue

            phi = np.abs(event["EMTFNtuple"].emtfTrack_phi[track]) * (np.pi / 180)
            eta = event["EMTFNtuple"].emtfTrack_eta[track]

            for GEM_hitref in GEM_hitrefs:
                # Put logic here for checking if this hit matches the eta and phi of the track within the dR cut self.dR_match_max
            
                if # I find that this GEM is within the dR cut
                    has_GEM_match[i] = True
        
        return good_tracks[has_GEM_match]
    


# me1theta = eval('evt_tree.emtfHit_emtf_theta[%d]' % (hitref))*pi/180
# Should be written as
# me1theta = event["EMTFNtuple"].emtfHit_emtf_theta[hitref]*pi/180