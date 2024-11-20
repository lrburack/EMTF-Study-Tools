from Dataset.Dataset import TrackSelector
from Dataset.constants import *

class MaxTrackSelector(TrackSelector):
    def __init__(self, mode: int, max_tracks: int, include_mode_15: bool = True):
        super().__init__(mode, include_mode_15)
        self.max_tracks = max_tracks
    
    # Return the valid track indexes
    def select(self, event):
        good_tracks = super().select(event)
        return good_tracks[:self.max_tracks]