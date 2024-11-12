import os
import ROOT
import numpy as np
from typing import List, Optional, Union, Tuple
from datetime import datetime

Run3TrainingVariables = {
    ## 4-station tracks

    # BASELINE mode 15 - dPhi12/23/34 + combos, theta, FR1, St1 ring, dTh14, bend1, RPC 1/2/3/4
    '15' : [
        'theta',
        'st1_ring2',
        'dPhi_12',
        'dPhi_23',
        'dPhi_34',
        'dPhi_13',
        'dPhi_14',
        'dPhi_24',
        'FR_1',
        'bend_1',
        'dPhiSum4',
        'dPhiSum4A',
        'dPhiSum3',
        'dPhiSum3A',
        'outStPhi',
        'dTh_14',
        'RPC_1',
        'RPC_2',
        'RPC_3',
        'RPC_4',
    ],

    ## 3-station tracks

    # BASELINE mode 14 - dPhi12/23/13, theta, FR1/2, St1 ring, dTh13, bend1, RPC 1/2/3
    '14' : [
        'theta',
        'st1_ring2',
        'dPhi_12',
        'dPhi_23',
        'dPhi_13',
        'FR_1',
        'FR_2',
        'bend_1',
        'dTh_13',
        'RPC_1',
        'RPC_2',
        'RPC_3',
    ],
    # BASELINE mode 13 - dPhi12/24/14, theta, FR1/2, St1 ring, dTh14, bend1, RPC 1/2/4
    '13' : [
        'theta',
        'st1_ring2',
        'dPhi_12',
        'dPhi_14',
        'dPhi_24',
        'FR_1',
        'FR_2',
        'bend_1',
        'dTh_14',
        'RPC_1',
        'RPC_2',
        'RPC_4',
    ],
    # BASELINE mode 11 - dPhi13/34/14, theta, FR1/3, St1 ring, dTh14, bend1, RPC 1/3/4
    '11' : [
        'theta',
        'st1_ring2',
        'dPhi_34',
        'dPhi_13',
        'dPhi_14',
        'FR_1',
        'FR_3',
        'bend_1',
        'dTh_14',
        'RPC_1',
        'RPC_3',
        'RPC_4',
    ],
    # BASELINE mode  7 - dPhi23/34/24, theta, FR2, dTh24, bend2, RPC 2/3/4
    '7' : [
        'theta',
        'dPhi_23',
        'dPhi_34',
        'dPhi_24',
        'FR_2',
        'bend_2',
        'dTh_24',
        'RPC_2',
        'RPC_3',
        'RPC_4',
    ],

    ## 2-station tracks

    # BASELINE mode 12 - dPhi12, theta, FR1/2, St1 ring, dTh12, bend1/2, RPC 1/2
    '12' : [
        'theta',
        'st1_ring2',
        'dPhi_12',
        'FR_1',
        'FR_2',
        'bend_1',
        'bend_2',
        'dTh_12',
        'RPC_1',
        'RPC_2',
    ],
    # BASELINE mode 10 - dPhi13, theta, FR1/3, St1 ring, dTh13, bend1/3, RPC 1/3
    '10' : [
        'theta',
        'st1_ring2',
        'dPhi_13',
        'FR_1',
        'FR_3',
        'bend_1',
        'bend_3',
        'dTh_13',
        'RPC_1',
        'RPC_3',
    ],
    # BASELINE mode  9 - dPhi14, theta, FR1/4, St1 ring, dTh14, bend1/4, RPC 1/4
    '9' : [
        'theta',
        'st1_ring2',
        'dPhi_14',
        'FR_1',
        'FR_4',
        'bend_1',
        'bend_4',
        'dTh_14',
        'RPC_1',
        'RPC_4',
    ],
    # BASELINE mode  6 - dPhi23, theta, FR2/3, dTh23, bend2/3, RPC 2/3
    '6' : [
        'theta',
        'dPhi_23',
        'FR_2',
        'FR_3',
        'bend_2',
        'bend_3',
        'dTh_23',
        'RPC_2',
        'RPC_3',
    ],
    # BASELINE mode  5 - dPhi24, theta, FR2/4, dTh24, bend2/4, RPC 2/4
    '5' : [
        'theta',
        'dPhi_24',
        'FR_2',
        'FR_4',
        'bend_2',
        'bend_4',
        'dTh_24',
        'RPC_2',
        'RPC_4',
    ],
    # BASELINE mode  3 - dPhi34, theta, FR3/4, dTh34, bend3/4, RPC 3/4
    '3' : [
        'theta',
        'dPhi_34',
        'FR_3',
        'FR_4',
        'bend_3',
        'bend_4',
        'dTh_34',
        'RPC_3',
        'RPC_4',
    ],
    # Null track, for testing EMTF performance
    '0' : [
        'theta',
        'RPC_3',
        'RPC_4',
    ],
}

#station-station transitions for delta phi's and theta's
TRANSITION_NAMES = ["12", "13", "14", "23", "24", "34"]

# TRANSITION_MAP[i] gives the stations corresponding to the transition index i
TRANSITION_MAP = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3]
])

# STATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions containing station i
STATION_TRANSITION_MAP = np.array([
    [0, 1, 2],
    [0, 3, 4],
    [1, 3, 5],
    [2, 4, 5]
])

# OUTSTATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions that don't contain station i
OUTSTATION_TRANSITION_MAP = np.array([
    [3, 4, 5],
    [1, 2, 5],
    [0, 2, 4],
    [0, 1, 3]
])

PRINT_EVENT = 100000

def get_station_presence(mode):
    return np.unpackbits(np.array([mode], dtype='>i8').view(np.uint8)).astype(bool)[-4:]

def stations(mode):
    return np.where(get_station_presence(mode))[0]

def transitions_from_mode(mode):
    # In the EMTFNtuple, dPhi and dTheta are 2D arrays. The first dimension is track. The second dimension is a 'transition index'
    # This transition index can be read as the index in this array: ["12", "13", "14", "23", "24", "34"]
    # For a particular mode, we want to know which of these transitions exists. This clever little array operation will get us this
    station_presence = get_station_presence(mode)
    return np.where(np.outer(station_presence, station_presence)[np.logical_not(np.tri(4))])[0]



# -------------------------------    Superclass Definitions    -----------------------------------
class TrainingVariable:
    def __init__(self, feature_names, tree_sources=[], feature_dependencies=[]):
        if isinstance(feature_names, list):
            self.feature_names = feature_names
        else:
            self.feature_names = [feature_names]

        self.tree_sources = tree_sources
        self.feature_dependencies = feature_dependencies

        # Assigned by Dataset class
        self.feature_inds = None
        self.shared_reference = None

    def configure(self):
        pass

    def calculate(self, event):
        pass

    def compress(self, event):
        pass

    # This can be overriden to configure variables for a mode. 
    # It must be overriden in the variable constructor takes input
    @classmethod
    def for_mode(cls, mode):
        return cls()


class EventFilter:
    def __init__(self):
        pass
    
    # Return true if the event should be kept, and false otherwise
    def filter(self, event, shared_info):
        return True

class HasModeFilter(EventFilter):
    def __init__(self):
        super().__init__()
        

    def filter(self, event, shared_info):
        return shared_info.track != None and event["EMTFNtuple"].emtfHit_size != 0

class RecoMatchFilter(EventFilter):
    def __init__(self, dR_match_max=0.4):
        self.dR_match_max = dR_match_max
        super().__init__()
    
    def filter(self, event, shared_info):
        return shared_info.reco_match != None

class SharedInfo:
    def __init__(self, mode, include_mode_15=True):
        self.mode = mode
        self.station_presence = get_station_presence(mode)
        self.stations = np.where(self.station_presence)[0] # note that these these are shifted to be zero indexed (-1 from station number)
        self.transition_inds = transitions_from_mode(mode)
        self.track = None
        self.hitrefs = np.zeros(len(self.station_presence), dtype=int)
        self.include_mode_15 = include_mode_15

        # Set by the Dataset constructor
        self.feature_names = None
        self.entry = None

    def calculate(self, event):
        self.track = None
        modes = np.array(event['EMTFNtuple'].emtfTrack_mode)
        if self.include_mode_15:
            good_track_inds = np.where((modes == self.mode) | (modes == 15))[0]
        else:
            good_track_inds = np.where((modes == self.mode))[0]
        
        if good_track_inds.size == 0:
            return
        
        self.track = int(good_track_inds[0])
        
        self.hitrefs = np.array([
            event['EMTFNtuple'].emtfTrack_hitref1[self.track],
            event['EMTFNtuple'].emtfTrack_hitref2[self.track],
            event['EMTFNtuple'].emtfTrack_hitref3[self.track],
            event['EMTFNtuple'].emtfTrack_hitref4[self.track],
        ], dtype=int)

    @classmethod
    def for_mode(cls, mode):
        return cls(mode)

class RecoSharedInfo(SharedInfo):
    def __init__(self, mode, dR_match_max=0.4):
        self.dR_match_max = dR_match_max
        self.reco_match = None
        super().__init__(mode)
    
    def calculate(self, event):
        self.track = None
        self.reco_match = None
        modes = np.array(event['EMTFNtuple'].emtfTrack_mode)
        good_track_inds = np.where((modes == self.mode) | (modes == 15))[0]
        if good_track_inds.size == 0:
            return
        self.track = int(good_track_inds[0])
        self.hitrefs = np.array([
            event['EMTFNtuple'].emtfTrack_hitref1[self.track],
            event['EMTFNtuple'].emtfTrack_hitref2[self.track],
            event['EMTFNtuple'].emtfTrack_hitref3[self.track],
            event['EMTFNtuple'].emtfTrack_hitref4[self.track],
        ], dtype=int)

        # We use the reco muon as the gen muon, but we must figure out which track in the EMTF corresponds to which gen muon
        # To do this we calculate dR=sqrt(dPhi^2 + dEta^2) using station 2 phi and eta, and find the minimum. 
        phi = np.abs(event["EMTFNtuple"].emtfUnpTrack_phi[self.track]) * (np.pi / 180)
        eta = event["EMTFNtuple"].emtfUnpTrack_eta[self.track]

        phi_diffs = np.abs(np.array(event["EMTFNtuple"].recoMuon_phiSt2)) - phi
        eta_diffs = np.array(event["EMTFNtuple"].recoMuon_etaSt2) - eta

        dR = np.sqrt(eta_diffs ** 2 + phi_diffs ** 2)

        match_ind = int(np.argmin(dR))
        if dR[match_ind] >= self.dR_match_max:
            return
        
        self.reco_match = match_ind

# -------------------------------    Dataset Definition    -----------------------------------
class Dataset:
    def __init__(self, variables: List[TrainingVariable], filters: Optional[List[EventFilter]] = [], shared_info: SharedInfo = None, compress: bool = False) -> 'Dataset':
        """
        Initializes the Dataset object with the specified training variables, filters, shared information, and compress option.

        :param variables: List of TrainingVariable objects defining the features used in the dataset.
        :param filters: Optional list of EventFilter objects for filtering the dataset. Defaults to an empty list.
        :param shared_info: SharedInfo object that provides shared information across different variables. If None, a new SharedInfo object is created.
        :param compress: Boolean indicating whether or not to compress the dataset. Defaults to False.
        """
        self.filters = filters
        self.variables = variables
        self.compress = compress

        self.feature_names = []
        for variable in self.variables:
            self.feature_names.extend(variable.feature_names)
            for feature in variable.feature_dependencies:
                if feature not in self.feature_names:
                    raise Exception(f"{type(variable)} is missing feature dependency {feature}. Current features: {self.feature_names}")
        self.num_features = len(self.feature_names)
        self.trainable_features = np.array([not feature_name.startswith("gen") for feature_name in self.feature_names], dtype='bool')

        self.entry = np.zeros(self.num_features, dtype='float32')

        if shared_info is None:
            self.shared_info = SharedInfo()
        else:
            self.shared_info = shared_info

        self.shared_info.feature_names = self.feature_names
        self.shared_info.entry_reference = self.entry

        start_ind = 0
        for variable in self.variables:
            variable.shared_reference = self.shared_info
            variable.feature_inds = self.entry[start_ind:start_ind + len(variable.feature_names)]
            variable.configure()
            start_ind += len(variable.feature_names)

        self.data = None
        self.filtered = None

    def apply_filters(self, event: dict, shared_info: SharedInfo) -> bool:
        """
        Applies all event filters to determine if an event should be kept or filtered out.

        :param event: The event data to be evaluated by the filters.
        :param shared_info: SharedInfo object that provides shared information to the filters.
        :return: True if the event passes all filters, False if it is filtered out.
        """
        for filter in self.filters:
            if not filter.filter(event, shared_info):
                return False
            
        return True

    def process_event(self, event: dict) -> np.ndarray:
        """
        Processes a single event by calculating the values for each variable and storing the results in the dataset's entry.

        :param event: The event data to be processed.
        :return: A numpy array representing the processed feature values for the event.
        """
        self.entry.fill(0)
        
        for i, variable in enumerate(self.variables):
            variable.calculate(event)

            if self.compress:
                variable.compress(event)

        return self.entry
    
    def build_dataset(self, raw_data: dict) -> np.ndarray:
        """
        Builds the dataset by processing events from the raw input data and applying filters.

        :param raw_data: A dictionary where keys are tree names and values are data trees (ROOT TChain objects).
        :return: A numpy array containing the processed data for all events that pass the filters. 
                 Rows corresponding to filtered events will contain only zeros.
        """
        tree_names = list(raw_data.keys())

        for variable in self.variables:
            for source in variable.tree_sources:
                if source not in tree_names:
                    raise Exception(f"{type(variable)} requires source {source} which is not present in the input data. Input data has {tree_names}")

        event_count = raw_data[tree_names[0]].GetEntries()
        for tree in tree_names[1:]:
            if raw_data[tree].GetEntries() != event_count:
                raise Exception("Different number of events in each tree")

        self.data = np.zeros((event_count, self.num_features), dtype='float32')
        self.filtered = np.zeros(event_count, dtype='bool')

        for event_num in range(event_count):
            if event_num % PRINT_EVENT == 0:
                print("* " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\t| Processing event #" + str(event_num))
                print("\t* Trainable event #" + str(np.sum(self.filtered)))
            
            for name in tree_names:
                raw_data[name].GetEntry(event_num)

            self.shared_info.calculate(raw_data)

            if not self.apply_filters(raw_data, self.shared_info):
                continue
            
            self.filtered[event_num] = True
            self.data[event_num] = self.process_event(raw_data)

        return self.data
    
    def get_features(self, features: Union[str, List[str]], filtered: bool = True) -> np.ndarray:
        """
        Retrieves the specified features from the dataset.

        :param features: A string or a list of feature names to retrieve.
        :param filtered: Boolean indicating whether to retrieve features from only filtered events (default is True).
        :return: A numpy array containing the requested features.
        """
        features = np.array([features]) if isinstance(features, str) else np.array(features) 
        for feature in features:
            if feature not in self.feature_names:
                raise Exception(f"{feature} is not a feature in this dataset.")
        
        if filtered:
            data = self.data[self.filtered]
        else:
            data = self.data
        
        return np.array(data[:, np.isin(np.array(self.feature_names), features)]).squeeze()

    def __str__(self) -> str:
        """
        Returns a string representation of the Dataset object, listing all feature names.

        :return: A string describing the Dataset object with its features.
        """
        return "Dataset object with features: " + ", ".join(self.feature_names)

    def randomize_event_order(self) -> None:
        """
        Randomizes the order of events in the dataset. This affects both the data and the filtered event mask.
        
        :return: None
        """
        permutation_inds = np.random.permutation(len(self.data))
        self.data = self.data[permutation_inds]
        self.filtered = self.filtered[permutation_inds]

    @staticmethod
    def get_root(base_dirs: List[str], files_per_endcap: int) -> Tuple[dict, List[str]]:
        """
        Builds a dictionary usable by the Dataset class from EMTFNtuple ROOT files. The dictionary key names correspond 
        to the names of trees (e.g., "EMTFNtuple", "MuShowerNtuple"), and each one contains a TChain object.

        :param base_dirs: List of base directories containing ROOT files.
        :param files_per_endcap: Maximum number of files to load for each endcap.
        :return: A tuple containing:
                 - A dictionary where the keys are tree names (e.g., "EMTFNtuple", "MuShowerNtuple"), 
                   and the values are TChain objects with concatenated trees.
                 - A list of file names that were successfully loaded.
        """
        event_data = {}
        file_names = []

        for base_dir in base_dirs:
            nFiles = 0
            break_loop = False
            
            for dirname, dirs, files in os.walk(base_dir):
                if break_loop: break
                for file in files:
                    if break_loop: break
                    if not file.endswith('.root'): continue
                    
                    file_name = os.path.join(dirname, file)
                    
                    root_file = ROOT.TFile.Open(file_name)
                    if not root_file or root_file.IsZombie():
                        print(f"Warning: Failed to open {file_name}")
                        continue
                    
                    nFiles += 1
                    print(f'* Loading file #{nFiles}: {file_name}')
                    file_names.append(file_name)

                    for key in root_file.GetListOfKeys():
                        obj = key.ReadObj()

                        if obj.InheritsFrom("TDirectory"):
                            dir_name = obj.GetName()

                            tree_name = "tree"
                            tree_chain_name = f"{dir_name}/{tree_name}"

                            if dir_name not in event_data:
                                event_data[dir_name] = ROOT.TChain(f"{tree_chain_name}")

                            event_data[dir_name].Add(f"{file_name}/{tree_chain_name}")

                    root_file.Close()

                    if nFiles >= files_per_endcap:
                        break_loop = True

        for dir_name, tchain in event_data.items():
            for branch in tchain.GetListOfBranches():
                branch_name = branch.GetName()
                if "Unp" in branch_name:
                    alias_name = branch_name.replace("Unp", "")
                    tchain.SetAlias(alias_name, branch_name)

        return event_data, file_names
    
    @classmethod
    def combine(cls, datasets: List['Dataset']) -> 'Dataset':
        """
        Combines multiple Dataset objects into a single Dataset object. The datasets must have the same number of events.

        :param datasets: A list of Dataset objects to be combined. All datasets must have the same number of events.
        :return: A new Dataset object that contains the variables and filters from all input datasets, and the combined data.
        """
        dataset_size = len(datasets[0].data)

        variables = []
        filters = []
        filtered = np.zeros(dataset_size, dtype='bool')
        for dataset in datasets:
            if len(dataset.data) != dataset_size:
                raise Exception("Datasets must have matching lengths")

            variables.extend(dataset.variables)
            filters.extend(dataset.filters)
            filtered = np.logical_or(filtered, dataset.filtered)
        
        new_dataset = cls(variables, filters = filters, shared_info=datasets[0].shared_info)
        new_dataset.filtered = filtered
        new_dataset.data = np.hstack([dataset.data for dataset in datasets])
        return new_dataset
    


# ughhhhhh

# -------------------------------    Training variable definitions    -----------------------------------
class GeneratorVariables(TrainingVariable):
    def __init__(self):
        super().__init__(["gen_pt", "gen_eta", "gen_phi"], tree_sources=["EMTFNtuple"])
    
    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].genPart_pt[0]
        self.feature_inds[1] = event['EMTFNtuple'].genPart_eta[0]
        self.feature_inds[2] = event['EMTFNtuple'].genPart_phi[0]    

class RecoVariables(GeneratorVariables):
    def __init__(self):
        super().__init__()
    
    def calculate(self, event):
        self.feature_inds[0] = event["EMTFNtuple"].recoMuon_pt[self.shared_reference.reco_match]
        self.feature_inds[1] = event["EMTFNtuple"].recoMuon_eta[self.shared_reference.reco_match]
        self.feature_inds[2] = event["EMTFNtuple"].recoMuon_phi[self.shared_reference.reco_match]
        
class TrackVariables(TrainingVariable):
    def __init__(self, branches):
        self.branches = branches
        super().__init__(branches, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for i, branch in enumerate(self.branches):
            self.feature_inds[i] = getattr(event["EMTFNtuple"], branch)[int(self.shared_reference.track)]

class HitVariables(TrainingVariable):
    def __init__(self, branches, stations):
        feature_names = []
        for branch in branches:
            for station in stations:
                feature_names += [branch + "_" + str(station + 1)]
        self.branches = branches
        self.stations = stations
        super().__init__(feature_names, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for i, branch in enumerate(self.branches):
            for j, station in enumerate(self.stations):
                ind = i * len(self.stations) + j
                self.feature_inds[ind] = getattr(event["EMTFNtuple"], branch)[int(self.shared_reference.hitrefs[station])]

class Theta(TrainingVariable):
    def __init__(self, theta_station):
        super().__init__("theta", tree_sources=["EMTFNtuple"])
        self.theta_station = theta_station
        
    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[self.theta_station])]

    def compress(self, event):
        theta = self.feature_inds[0]
        if self.shared_reference.mode == 15:
            if event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track] == 0:
                theta = (min(max(theta, 5), 52) - 5) / 6
            else:
                theta = ((min(max(theta, 46), 87) - 46) / 7) + 8
        else: 
            if event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track] == 0: 
                theta = (max(theta, 1) - 1) / 4
            else: 
                theta = ((min(theta, 104) - 1) / 4) + 6
        self.feature_inds[0] = int(theta)

    @classmethod
    def for_mode(cls, mode):
        # The theta value we use is always the theta in the first station that is not station 1.
        station_presence = get_station_presence(mode)
        return cls(np.argmax(station_presence[1:]) + 1)


class St1_Ring2(TrainingVariable):
    def __init__(self):
        super().__init__("st1_ring2", tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track]


class dPhi(TrainingVariable):
    # Used for compression
    NLBMap_4bit_256Max = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 31, 46, 68, 136]
    NLBMap_5bit_256Max = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136]
    NLBMap_7bit_512Max = [
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
        22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
        44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  79,  80,  81,  83,  84,  86,  87,  89,  91,  92,  94,
        96,  98,  100, 102, 105, 107, 110, 112, 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168,
        174, 181, 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470]

    def __init__(self, transition_inds):
        self.transition_inds = transition_inds
        # dPhi is defined by two stations, so which dPhi's we train on depends on the mode
        features = ["dPhi_" + TRANSITION_NAMES[ind] for ind in transition_inds]
        super().__init__(features, tree_sources=["EMTFNtuple"])

        self.nBitsA = 7
        self.nBitsB = 7
        self.nBitsC = 7
        self.maxA = 512
        self.maxB = 512
        self.maxC = 512

    def configure(self):
        mode = self.shared_reference.mode
        if mode == 7 or mode == 11 or mode > 12:
            self.nBitsB = 5
            self.maxB = 256
            self.nBitsC = 5
            self.maxC = 256
        if mode == 15:
            self.nBitsC = 4
            self.maxC = 256

    def calculate(self, event):
        # for feature_ind, transition_ind in enumerate(self.transition_inds):
        #     sign = 1 if event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)][int(transition_ind)] else -1
        #     self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfTrack_ptLUT_deltaPh[int(self.shared_reference.track)][int(transition_ind)] * sign
        # Convert ROOT data to NumPy arrays (if they aren't already NumPy arrays)
        signs = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)])
        deltaPh = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_deltaPh[int(self.shared_reference.track)])
        
        # Convert signs to -1 or 1 using NumPy's where function
        signs = np.where(signs[self.transition_inds], 1, -1)
        
        # Multiply deltaPh by sign using NumPy array operations
        self.feature_inds[:] = deltaPh[self.transition_inds] * signs * signs[0]
    
    def compress(self, event):
        dphi = 0
        mode = self.shared_reference.mode

        for feature_ind, transition_ind in enumerate(self.transition_inds):
            dphi = self.feature_inds[feature_ind]
            if transition_ind == 3:
                if mode == 7:
                    dphi = dPhi.getNLBdPhi(dphi, self.nBitsA, self.maxA)
                else:
                    dphi = dPhi.getNLBdPhi(dphi, self.nBitsB, self.maxB)
            elif transition_ind == 4:
                dphi = dPhi.getNLBdPhi(dphi, self.nBitsB, self.maxB)
            elif transition_ind == 5:
                dphi = dPhi.getNLBdPhi(dphi, self.nBitsC, self.maxC)
            else:
                dphi = dPhi.getNLBdPhi(dphi, self.nBitsA, self.maxA)

            self.feature_inds[feature_ind] = dphi

        if mode == 15:
            self.feature_inds[1] = self.feature_inds[0] + self.feature_inds[3]
            self.feature_inds[2] = self.feature_inds[1] + self.feature_inds[5]
            self.feature_inds[4] = self.feature_inds[3] + self.feature_inds[5]
        elif mode == 14:
            self.feature_inds[1] = self.feature_inds[0] + self.feature_inds[2]
        elif mode == 13:
            self.feature_inds[1] = self.feature_inds[0] + self.feature_inds[2]
        elif mode == 11:
            self.feature_inds[1] = self.feature_inds[0] + self.feature_inds[2]
        elif mode == 7:
            self.feature_inds[1] = self.feature_inds[0] + self.feature_inds[2]

    @classmethod
    def for_mode(cls, mode):
        return cls(transitions_from_mode(mode))
    
    @classmethod
    def getNLBdPhi(cls, dphi, bits, max):
        dphi_ = max
        sign_ = 1
        if dphi < 0:
            sign_ = -1
        dphi = sign_ * dphi

        if max == 256:
            if bits == 4:
                dphi_ = dPhi.NLBMap_4bit_256Max[(1 << bits) - 1]
                for edge in range ((1 << bits) - 1):
                    if dPhi.NLBMap_4bit_256Max[edge] <= dphi and dPhi.NLBMap_4bit_256Max[edge + 1] > dphi:
                        dphi_= dPhi.NLBMap_4bit_256Max[edge]
            if bits == 5:
                dphi_ = dPhi.NLBMap_5bit_256Max[(1 << bits) - 1]
                for edge in range((1 << bits) - 1):
                    if dPhi.NLBMap_5bit_256Max[edge] <= dphi and dPhi.NLBMap_5bit_256Max[edge + 1] > dphi:
                        dphi_ = dPhi.NLBMap_5bit_256Max[edge]
        elif max == 512:
            if bits == 7:
                dphi_ = dPhi.NLBMap_7bit_512Max[(1 << bits) - 1]
                for edge in range((1 << bits) - 1):
                    if dPhi.NLBMap_7bit_512Max[edge] <= dphi and dPhi.NLBMap_7bit_512Max[edge + 1] > dphi:
                        dphi_ = dPhi.NLBMap_7bit_512Max[edge]


        return sign_ * dphi_


class dTh(TrainingVariable):
    # Use the dTh between the furthest apart stations.
    mode_transition_map = {
        15: [2],  # Mode 15 uses dTh_14, so transition 14 (index 2)
        14: [1],  # Mode 14 uses dTh_13, so transition 13 (index 1)
        13: [2],  # Mode 13 uses dTh_14, so transition 14 (index 2)
        11: [2],  # Mode 11 uses dTh_14, so transition 14 (index 2)
        7: [4],   # Mode 7 uses dTh_24, so transition 24 (index 4)
        12: [0],  # Mode 12 uses dTh_12, so transition 12 (index 0)
        10: [1],  # Mode 10 uses dTh_13, so transition 13 (index 1)
        9: [2],   # Mode 9 uses dTh_14, so transition 14 (index 2)
        6: [3],   # Mode 6 uses dTh_23, so transition 23 (index 3)
        5: [4],   # Mode 5 uses dTh_24, so transition 24 (index 4)
        3: [5],   # Mode 3 uses dTh_34, so transition 34 (index 5)
        0: [None] # Mode 0 doesn't use any dTh transitions
    }

    def __init__(self, transition_inds):
        self.transition_inds = transition_inds
        # dTh is defined by two stations, so which dTh's we train on depends on the mode
        features = ["dTh_" + TRANSITION_NAMES[ind] for ind in transition_inds]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for feature_ind, transition_ind in enumerate(self.transition_inds):
            self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[TRANSITION_MAP[transition_ind][1]])] - event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[TRANSITION_MAP[transition_ind][0]])]
        # Convert the hitrefs and theta values to NumPy arrays
        # hitrefs = self.shared_reference.hitrefs
        # theta_vals = np.array(event['EMTFNtuple'].emtfHit_emtf_theta)
        
        # # Use vectorized NumPy operations to calculate the dTheta for all transitions
        # self.feature_inds[:] = theta_vals[hitrefs[TRANSITION_MAP[self.transition_inds, 1]]] - theta_vals[hitrefs[TRANSITION_MAP[self.transition_inds, 0]]]

    def compress(self, event):
        for feature_ind, transition_ind in enumerate(self.transition_inds):
            dTheta = self.feature_inds[feature_ind]
            if self.shared_reference.mode == 15:
                if abs(dTheta) <= 1:
                    dTheta = 2
                elif abs(dTheta) <= 2:
                    dTheta = 1
                elif dTheta <= -3:
                    dTheta = 0
                else:
                    dTheta = 3 
            else:
                if dTheta <= -4:
                    dTheta = 0
                elif -3 <= dTheta <= 2 : dTheta += 4
                else: dTheta = 7

            self.feature_inds[feature_ind] = dTheta


    @classmethod
    def for_mode(cls, mode):
        return cls(dTh.mode_transition_map[mode])


class FR(TrainingVariable):
    # Which stations to use depending on the mode
    mode_station_map = {
        15: [0],
        14: [0, 1],
        13: [0, 1],
        11: [0, 2],
        7: [1],
        12: [0, 1],
        10: [0, 2],
        9: [0, 3],
        6: [1, 2],
        5: [1, 3],
        3: [2, 3],
        0: [2, 3],
    }

    def __init__(self, stations):
        self.stations = stations
        features = ["FR_" + str(station + 1) for station in stations]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for feature_ind, station in enumerate(self.stations):
            self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfTrack_ptLUT_fr[int(self.shared_reference.track)][int(station)]

    @classmethod
    def for_mode(cls, mode):
        return cls(FR.mode_station_map[mode])


class RPC(TrainingVariable):
    def __init__(self, stations):
        self.stations = stations
        self.theta_ind = None
        features = ["RPC_" + str(station + 1) for station in stations]
        super().__init__(features, feature_dependencies=["theta"], tree_sources=["EMTFNtuple"])
    
    def configure(self):
        self.theta_ind = self.shared_reference.feature_names.index("theta")

    def calculate(self, event):
       # Convert the cpattern values to a NumPy array
        cpattern_vals = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track])
        # Use vectorized NumPy operations to check if the pattern is zero (indicating RPC use)
        self.feature_inds[:] = cpattern_vals[self.stations] == 0

    def compress(self, event):
        mode = self.shared_reference.mode
        if mode == 15 and event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track] == 0:
            if self.shared_reference.entry_reference[self.theta_ind] < 4:
                self.feature_inds[2] = 0
                self.feature_inds[3] = 0

        # The logic after this has to do with removing redundant RPC information if more than
        # one RPC was used in building the track. We dont need to do this if there are less than 2 RPCs used
        if np.sum(self.feature_inds) < 2:
            return

        # Its convenient to have this as a bool array
        rpc = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track])[self.stations] == 0

        if mode == 15:
            if rpc[0] and rpc[1]:
                self.feature_inds[2] = 0
                self.feature_inds[3] = 0
            elif rpc[0] and rpc[2]:
                self.feature_inds[3] = 0
            elif rpc[3] and rpc[1]:
                self.feature_inds[2] = 0
            elif rpc[2] and rpc[3] and event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track] == 0:
                self.feature_inds[2] = 0
        else: # For 3 station only
            # If the first RPC is present we dont care about the others
            if rpc[0]:
                self.feature_inds[1] = 0
                self.feature_inds[2] = 0
            elif rpc[2]:
                self.feature_inds[1] = 0

    @classmethod
    def for_mode(cls, mode):
        # There should be an RPC feature for each station in the track
        station_presence = get_station_presence(mode)
        return cls(np.where(station_presence)[0])


class Bend(TrainingVariable):
    # Which stations to use depending on the mode
    mode_station_map = {
        15: [0],
        14: [0],
        13: [0],
        11: [0],
        7: [1],
        12: [0, 1],
        10: [0, 2],
        9: [0, 3],
        6: [1, 2],
        5: [1, 3],
        3: [2, 3],
        0: [],
    }

    pattern_bend_map = np.array([0, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0], dtype='float32')

    def __init__(self, stations):
        self.stations = stations
        features = ["bend_" + str(station + 1) for station in stations]
        feature_dependencies = ["RPC_" + str(station + 1) for station in stations]
        super().__init__(features, feature_dependencies=feature_dependencies, tree_sources=["EMTFNtuple"])
        
        # Set by configure
        self.nBits = None
        self.RPC_inds = None

    def configure(self):
        mode = self.shared_reference.mode
        self.nBits = 2 if mode == 7 or mode == 11 or mode > 12 else 3

        self.RPC_inds = np.array([self.shared_reference.feature_names.index("RPC_" + str(station + 1)) for station in self.stations])

    def calculate(self, event):
        cpattern_vals = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track])[self.stations]
        self.feature_inds[:] = Bend.pattern_bend_map[cpattern_vals] * -1 * event['EMTFNtuple'].emtfTrack_endcap[self.shared_reference.track]

    def compress(self, event):
        nBits = self.nBits

        signs = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)])        
        signs = np.where(signs[self.shared_reference.transition_inds], 1, -1)
        sign_ = event['EMTFNtuple'].emtfTrack_endcap[self.shared_reference.track] * -1 * signs[0]

        for feature_ind, station in enumerate(self.stations):
            pattern = event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track][station]

            if nBits == 2:
                if pattern == 10:
                    clct_ = 1
                elif pattern == 9:
                    clct_ = 1 if sign_ > 0 else 2
                elif pattern == 8:
                    clct_ = 2 if sign_ > 0 else 1
                elif pattern == 7:
                    clct_ = 0 if sign_ > 0 else 3
                elif pattern == 6:
                    clct_ = 3 if sign_ > 0 else 0
                elif pattern == 5:
                    clct_ = 0 if sign_ > 0 else 3 
                elif pattern == 4:
                    clct_ = 3 if sign_ > 0 else 0
                elif pattern == 3:
                    clct_ = 0 if sign_ > 0 else 3
                elif pattern == 2:
                    clct_ = 3 if sign_ > 0 else 0
                elif pattern == 1:
                    clct_ = 0 if sign_ > 0 else 3
                elif pattern == 0 and not self.shared_reference.entry_reference[self.RPC_inds[feature_ind]] == 1:
                    clct_ = 0
                else:
                    clct_ = 1
            elif nBits == 3:
                if pattern == 10:
                    clct_ = 4
                elif pattern == 9:
                    clct_ = 3 if sign_ > 0 else 5
                elif pattern == 8:
                    clct_ = 5 if sign_ > 0 else 3
                elif pattern == 7:
                    clct_ = 2 if sign_ > 0 else 6
                elif pattern == 6:
                    clct_ = 6 if sign_ > 0 else 2
                elif pattern == 5:
                    clct_ = 1 if sign_ > 0 else 7
                elif pattern == 4:
                    clct_ = 7 if sign_ > 0 else 1
                elif pattern == 3:
                    clct_ = 1 if sign_ > 0 else 7
                elif pattern == 2:
                    clct_ = 7 if sign_ > 0 else 1
                elif pattern == 1:
                    clct_ = 1 if sign_ > 0 else 7
                elif pattern == 0:
                    clct_ = 0
                else:
                    clct_ = 4
            self.feature_inds[feature_ind] = clct_

    @classmethod
    def for_mode(cls, mode):
        return Bend(Bend.mode_station_map[mode])

# Classes which extend this will be able to access already calculated dPhi's
class dPhiSum(TrainingVariable):
    def __init__(self, transitions, feature_name="dPhiSum"):
        feature_dependencies = ["dPhi_" + str(transition) for transition in transitions]
        super().__init__(feature_name, feature_dependencies=feature_dependencies)
        self.transitions = transitions
        self.dPhi_reference_inds = None

    def configure(self):
        self.dPhi_reference_inds = np.array([self.shared_reference.feature_names.index("dPhi_" + str(transition)) for transition in self.transitions])

    def calculate(self, event):
        self.feature_inds[0] = np.sum(self.shared_reference.entry_reference[self.dPhi_reference_inds])

    @classmethod
    def for_mode(cls, mode):
        return cls(transitions_from_mode(mode))

class dPhiSumA(TrainingVariable):
    def __init__(self, transitions, feature_name="dPhiSumA"):
        feature_dependencies = ["dPhi_" + str(transition) for transition in transitions]
        super().__init__(feature_name, feature_dependencies=feature_dependencies)
        self.transitions = transitions
        self.dPhi_reference_inds = None

    def configure(self):
        self.dPhi_reference_inds = np.array([self.shared_reference.feature_names.index("dPhi_" + str(transition)) for transition in self.transitions])

    def calculate(self, event):
        self.feature_inds[0] = np.sum(np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds]))

    @classmethod
    def for_mode(cls, mode):
        return cls(transitions_from_mode(mode))

# -------------------------------    For use with mode 15    -----------------------------------
class dPhiSum4(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum4")

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum4 is for mode 15 only")
        return cls()


class dPhiSum4A(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum4A")
    
    def calculate(self, event):
        self.feature_inds[0] = np.sum(np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds]))

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum4A is for mode 15 only")
        return cls()


class OutStPhi(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="outStPhi")
        self.station_deviations = np.zeros(4, dtype='float32')

    def calculate(self, event):
        dPhis = np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds])
        self.station_deviations[:] = np.sum(dPhis[STATION_TRANSITION_MAP], axis=1)
        max_deviation = np.max(self.station_deviations) == self.station_deviations

        if np.sum(max_deviation) > 1:
            self.feature_inds[0] = -1
        else:
            self.feature_inds[0] = np.argmax(max_deviation)

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3 is for mode 15 only")
        return cls()


class OutStPhi_ShowerInformed(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="outStPhi")
        self.station_deviations = np.zeros(4, dtype='float32')

    def calculate(self, event):
        if np.any(self.shared_reference.showers_on_track):
            rightmost_true_col = np.max(np.where(self.shared_reference.showers_on_track)[1])
            rows_with_true_in_col = np.where(self.shared_reference.showers_on_track[:, rightmost_true_col])[0]
            self.feature_inds[0] = rows_with_true_in_col[0]
        else:
            dPhis = np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds])
            self.station_deviations[:] = np.sum(dPhis[STATION_TRANSITION_MAP], axis=1)
            max_deviation = np.max(self.station_deviations) == self.station_deviations

            if np.sum(max_deviation) > 1:
                self.feature_inds[0] = -1
            else:
                self.feature_inds[0] = np.argmax(max_deviation)

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3 is for mode 15 only")
        return cls()


class dPhiSum3(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum3")
        self.feature_dependencies.extend(["outStPhi"])
        self.out_station_phi_reference = None
    
    def configure(self):
        super().configure()
        self.out_station_phi_reference = self.shared_reference.feature_names.index("outStPhi")

    def calculate(self, event):
        out_station = int(self.shared_reference.entry_reference[self.out_station_phi_reference])
        if out_station == -1:
            out_station = 0
        self.feature_inds[0] = np.sum(
            self.shared_reference.entry_reference[self.dPhi_reference_inds[
                OUTSTATION_TRANSITION_MAP[out_station]
                ]])
    
    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3A is for mode 15 only")
        return cls()


class dPhiSum3A(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum3A")
        self.feature_dependencies.extend(["outStPhi"])
        self.out_station_phi_reference = None
    
    def configure(self):
        super().configure()
        self.out_station_phi_reference = self.shared_reference.feature_names.index("outStPhi")

    def calculate(self, event):
        out_station = int(self.shared_reference.entry_reference[self.out_station_phi_reference])
        if out_station == -1:
            out_station = 0
        self.feature_inds[0] = np.sum(np.abs(
            self.shared_reference.entry_reference[self.dPhi_reference_inds[
                OUTSTATION_TRANSITION_MAP[out_station]
                ]]))
    
    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3 is for mode 15 only")
        return cls()

# -------------------------------    Shower Stuff    -----------------------------------
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
