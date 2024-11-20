from Dataset.Dataset import TrainingVariable
from Dataset.constants import *

class ShowerBit(TrainingVariable):
    def __init__(self, shower_threshold = 1):
        super().__init__(feature_names="shower_bit_thresh=" + str(shower_threshold), tree_sources=['MuShowerNtuple'])
        self.shower_threshold = shower_threshold

    def calculate(self, event):
        self.feature_inds[0] = event['MuShowerNtuple'].CSCShowerDigiSize >= self.shower_threshold


class AllShower(TrainingVariable):
    match_by_valid = ["chamber", "sector"]
    # Transate a chamber to a sector
    all_sector_map = np.roll((np.arange(36) // 6) + 1, 2)
    outer_station_ring1_sector_map = np.roll((np.arange(18) // 3) + 1, 1)

    # Find the neightbor sector of a chamber
    all_neighbor_sector_map = np.array([-1, 1, 6, -1, -1, -1, -1, 2, 1, -1, -1, -1, -1, 3, 2, -1, -1, -1, -1, 4, 3, -1 , -1, -1, -1, 5, 4, -1, -1, -1, -1, 6, 5, -1, -1, -1])
    outer_station_ring1_neighbor_sector_map = np.array([1, 6, -1, 2, 1, -1, 3, 2, -1, 4, 3, -1, 5, 4, -1, 6, 5, -1])

    def __init__(self, stations, match_by = "chamber"):
        if match_by not in AllShower.match_by_valid:
            ValueError(f"Cannot match by {match_by}. Options are: {AllShower.match_by_valid}")

        feature_names = []
        for station in stations + 1:
            feature_names.extend(["loose_" + str(station), "nominal_" + str(station), "tight_" + str(station)])
        super().__init__(feature_names=feature_names, tree_sources=['EMTFNtuple','MuShowerNtuple'])

        self.match_by = AllShower.match_by_valid.index(match_by)
        self.stations = stations
        self.showers_on_track = np.zeros((4, 3), dtype='bool')
    
    def configure(self):
        self.shared_reference.showers_on_track = self.showers_on_track
    
    def calculate(self, event):
        # This code block matches showers with hits along the track
        # This array contains the shower information. The first axis corresponds to the station, the second axis corresponds to the shower type (0: loose, 1: nominal, 2: tight)
        # Loop through each hit. We will check for a corresponding shower
        self.showers_on_track.fill(0)

        for station in self.stations:
            hitref = int(self.shared_reference.hitrefs[station])

            # Loop through each shower and see if it corresponds to a hit in the track
            for i in range(event['MuShowerNtuple'].CSCShowerDigiSize):
                # Check that the hit location matches the shower location
                if self.match_by == 0: # Match by chamber
                    if (event['EMTFNtuple'].emtfHit_chamber[hitref] == event['MuShowerNtuple'].CSCShowerDigi_chamber[i] and 
                        event['EMTFNtuple'].emtfHit_ring[hitref] == event['MuShowerNtuple'].CSCShowerDigi_ring[i] and 
                        event['EMTFNtuple'].emtfHit_station[hitref] == event['MuShowerNtuple'].CSCShowerDigi_station[i] and 
                        event['EMTFNtuple'].emtfHit_endcap[hitref] == event['MuShowerNtuple'].CSCShowerDigi_endcap[i]):
                        # Add the shower information to the array
                        self.showers_on_track[station, :] = np.array([int(event['MuShowerNtuple'].CSCShowerDigi_oneLoose[i]), int(event['MuShowerNtuple'].CSCShowerDigi_oneNominal[i]), int(event['MuShowerNtuple'].CSCShowerDigi_oneTight[i])]).T
                if self.match_by == 1: # Match by sector
                    if event['MuShowerNtuple'].CSCShowerDigi_ring[i] == 1 and event['MuShowerNtuple'].CSCShowerDigi_station[i] > 1:
                        sector = AllShower.outer_station_ring1_sector_map[event['MuShowerNtuple'].CSCShowerDigi_chamber[i] - 1]
                        neighbor_sector = AllShower.outer_station_ring1_neighbor_sector_map[event['MuShowerNtuple'].CSCShowerDigi_chamber[i] - 1]
                    else:
                        sector = AllShower.all_sector_map[event['MuShowerNtuple'].CSCShowerDigi_chamber[i] - 1]
                        neighbor_sector = AllShower.all_neighbor_sector_map[event['MuShowerNtuple'].CSCShowerDigi_chamber[i] - 1]
                    
                    if ((event['EMTFNtuple'].emtfHit_sector[hitref] == sector or event['EMTFNtuple'].emtfHit_sector[hitref] == neighbor_sector) and
                        event['EMTFNtuple'].emtfHit_station[hitref] == event['MuShowerNtuple'].CSCShowerDigi_station[i] and
                        event['EMTFNtuple'].emtfHit_endcap[hitref] == event['MuShowerNtuple'].CSCShowerDigi_endcap[i]):
                        self.showers_on_track[station, :] = np.logical_or(
                            self.showers_on_track[station, :],
                            np.array([int(event['MuShowerNtuple'].CSCShowerDigi_oneLoose[i]), int(event['MuShowerNtuple'].CSCShowerDigi_oneNominal[i]), int(event['MuShowerNtuple'].CSCShowerDigi_oneTight[i])]).T
                        )
                        
        for feature_num, station in enumerate(self.stations):
            self.feature_inds[3 * feature_num : 3 * feature_num + 3] = self.showers_on_track[station, :]
    
    @classmethod
    def for_mode(cls, mode):
        return cls(np.where(get_station_presence(mode))[0])


class ShowerCount(TrainingVariable):
    valid_shower_types = ["loose", "nominal", "tight"]

    def __init__(self, stations, shower_types):
        for type in shower_types:
            if type not in ShowerCount.valid_shower_types:
                raise Exception(type + " is not a valid kind of shower")
        feature_names = [shower_type + "_showerCount" for shower_type in shower_types]
        super().__init__(feature_names=feature_names, tree_sources=['EMTFNtuple','MuShowerNtuple'])
        self.stations = stations
        self.shower_types = np.isin(np.array(ShowerCount.valid_shower_types), np.array(shower_types))
    
    def calculate(self, event):
        # print("according to ShowerCount: " + str(self.shared_reference.showers_on_track))
        self.feature_inds[:] = np.sum(self.shared_reference.showers_on_track, axis=0)[self.shower_types]

class CarefulShowerBit(TrainingVariable):
    def __init__(self, stations, shower_threshold = 1):
        super().__init__(feature_names=["careful_shower_bit_thresh=" + str(shower_threshold)], tree_sources=['EMTFNtuple','MuShowerNtuple'])
        self.stations = stations
        self.shower_threshold = shower_threshold
    
    def calculate(self, event):          
        self.feature_inds[0] = np.sum(self.shared_reference.showers_on_track[:, 0]) >= self.shower_threshold

    @classmethod
    def for_mode(cls, mode):
        return cls(np.where(get_station_presence(mode))[0])


class ShowerStationType(TrainingVariable):
    def __init__(self, stations):
        feature_names = ["shower_type_" + str(station) for station in stations]
        super().__init__(feature_names=feature_names, tree_sources=['EMTFNtuple','MuShowerNtuple'])
        self.stations = stations
    
    def calculate(self, event):          
        self.feature_inds[:] = np.sum(self.shared_reference.showers_on_track[self.stations, :], axis=1)

    @classmethod
    def for_mode(cls, mode):
        return cls(np.where(get_station_presence(mode))[0])
