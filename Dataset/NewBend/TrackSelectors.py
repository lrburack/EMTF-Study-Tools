from Dataset.Dataset import TrackSelector
from Dataset.constants import *
from Dataset.NewBend import NewBend_LUT

from Dataset.NewBend.NewBend_LUT import CSCGEMSlopecorrector


class NewBendTrackSelector(TrackSelector):
    # This is made only for modes which contain station 1.
    def __init__(self, mode: int, include_mode_15: bool = True, dR_match_max: float = 15):
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

        # Candidate GEMs
        GEM_hitrefs = np.where(np.array(event["EMTFNtuple"].emtfHit_quality) >= 4)[0]
        if len(GEM_hitrefs) > 0:
            print(len(GEM_hitrefs))

        # Loop through each track which was identified as 'good' and check if it has a matching GEM hit
        for i, track in enumerate(good_tracks):
            # Remove station 1 ring 2 because this cant have a GEM match
            if event["EMTFNtuple"].emtfTrack_ptLUT_st1_ring2[int(track)]:
                continue

            # Get information about the track which we will use to calculate dR
            station_1_hitref = event["EMTFNtuple"].emtfTrack_hitref1[int(track)]
            me1phi = event["EMTFNtuple"].emtfHit_emtf_phi[station_1_hitref]
            me1theta = event["EMTFNtuple"].emtfHit_emtf_theta[station_1_hitref]
            slope = event["EMTFNtuple"].emtfHit_slope[station_1_hitref]
            slope_sign = 1 if event["EMTFNtuple"].emtfHit_bend[station_1_hitref] % 2 == 0 else -1 # Odd bend means negative slope
            isME1a = event["EMTFNtuple"].emtfHit_ring[station_1_hitref] == 4    # Ring 4 means ME1a

            # See if there is a GEM hit within the dR cut of the station 1 hit
            for GEM_hitref in GEM_hitrefs:
                GEM_hitref = int(GEM_hitref)

                if event["EMTFNtuple"].emtfHit_endcap[GEM_hitref] != event["EMTFNtuple"].emtfHit_endcap[station_1_hitref]:
                    continue

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

                # print(f"dphi_corr: {dphi_corr}, dphi: {dphi}, dtheta: {dtheta}, dR: {dR}")
                # print(f"\tme1phi: {me1phi}, ge1phi: {event['EMTFNtuple'].emtfHit_emtf_phi[GEM_hitref]}")
                # print(f"\tme1theta: {me1theta}, ge1theta: {event['EMTFNtuple'].emtfHit_emtf_theta[GEM_hitref]}")

                if dR < self.dR_match_max: # If they match, we can keep this track and skip to checking the next one
                    has_GEM_match[i] = True
                    break

        
        return good_tracks[has_GEM_match]
    


# me1theta = eval('evt_tree.emtfHit_emtf_theta[%d]' % (hitref))*pi/180
# Should be written as
# me1theta = event["EMTFNtuple"].emtfHit_emtf_theta[hitref]*pi/180