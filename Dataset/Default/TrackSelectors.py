from Dataset.Dataset import TrackSelector
from Dataset.constants import *

class MaxTrackSelector(TrackSelector):
    def __init__(self, mode: int, max_tracks: int, include_mode_15: bool = True, tracks_per_endcap=None):
        super().__init__(mode, include_mode_15=include_mode_15, tracks_per_endcap=tracks_per_endcap)
        self.max_tracks = max_tracks
    
    # Return the valid track indexes
    def select(self, event):
        good_tracks = super().select(event)
        return good_tracks[:self.max_tracks]
    
# Like nick hurley's old code. Take only one track, but dont impose that its in the central bx
class OldTrackSelector(TrackSelector):
    def __init__(self, mode: int, include_mode_15: bool = True, tracks_per_endcap=None):
        super().__init__(mode, include_mode_15=include_mode_15, tracks_per_endcap=tracks_per_endcap)
    
    # Return the valid track indexes
    def select(self, event):
        modes = np.array(event["EMTFNtuple"].emtfTrack_mode)
        if self.include_mode_15:
            return np.where((modes == self.mode) | (modes == 15))[0][:1]
        else:
            return np.where((modes == self.mode))[0][:1]
        
# Used to get true efficiency plots where we count all the gen muons! -- Dont try to use any TrainingVariables except GeneratorVariables with this
class AllEventSelector(TrackSelector):
    def __init__(self):
        pass
    
    def select(self, event):
        return np.zeros(1)