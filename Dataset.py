import os
import ROOT
import numpy as np
from typing import List, Optional, Union, Tuple
from datetime import datetime
from constants import *

PRINT_EVENT = 100000

# -------------------------------    Superclass Definitions    -----------------------------------
class TrainingVariable:
    def __init__(self, feature_names, tree_sources=[], feature_dependencies=[], trainable: bool = True):
        if isinstance(feature_names, list):
            self.feature_names = feature_names
        else:
            self.feature_names = [feature_names]

        self.tree_sources = tree_sources
        self.feature_dependencies = feature_dependencies
        self.trainable = trainable

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


class TrackSelector:
    def __init__(self, mode: int, include_mode_15: bool = True):
        self.mode = mode
        self.include_mode_15 = include_mode_15
    
    # Return the valid track indexes
    def select(self, event):
        modes = np.array(event["EMTFNtuple"].emtfTrack_mode)
        if self.include_mode_15:
            return np.where((modes == self.mode) | (modes == 15))[0]
        else:
            return np.where(modes == self.mode)[0]


class SharedInfo:
    # This class contains information that is generally needed by all of the variables to do their calculation.
    # Some of the information is constant for an entire dataset (the mode etc.) and some is updated every event by calculate()
    # All variables have a reference to this class and use its information
    def __init__(self, mode: int):
        self.mode = mode
        self.station_presence = get_station_presence(mode)
        self.stations = stations(mode) # note that these these are shifted to be zero indexed (-1 from station number)
        self.transition_inds = transitions_from_mode(mode)

        # Set by calculate()
        self.track = None
        self.hitrefs = np.zeros(len(self.station_presence), dtype=int)

        # Set by the Dataset constructor
        self.feature_names = None
        self.entry = None

    def calculate(self, event, track):
        print(track)
        self.track = int(track)
        
        self.hitrefs[0] = int(event['EMTFNtuple'].emtfTrack_hitref1[self.track])
        self.hitrefs[1] = int(event['EMTFNtuple'].emtfTrack_hitref2[self.track])
        self.hitrefs[2] = int(event['EMTFNtuple'].emtfTrack_hitref3[self.track])
        self.hitrefs[3] = int(event['EMTFNtuple'].emtfTrack_hitref4[self.track])

    @classmethod
    def for_mode(cls, mode):
        return cls(mode)

# -------------------------------    Dataset Definition    -----------------------------------
class Dataset:
    def __init__(self, variables: List[TrainingVariable], track_selector: TrackSelector = None, shared_info: SharedInfo = None, compress: bool = False) -> 'Dataset':
        """
        Initializes the Dataset object with the specified training variables, filters, shared information, and compress option.

        :param variables: List of TrainingVariable objects defining the features used in the dataset.
        :param filters: Optional list of EventFilter objects for filtering the dataset. Defaults to an empty list.
        :param shared_info: SharedInfo object that provides shared information across different variables. If None, a new SharedInfo object is created.
        :param compress: Boolean indicating whether or not to compress the dataset. Defaults to False.
        """
        self.track_selector = track_selector
        self.variables = variables
        self.compress = compress

        self.feature_names = []
        self.trainable_features = [] # This will be a boolean array with True where the variable says its features are "trainable" (GeneratorVariables, for example, will set this to false)
        for variable in self.variables:
            self.feature_names.extend(variable.feature_names)
            self.trainable_features.extend([variable.trainable] * len(variable.feature_names))
            for feature in variable.feature_dependencies:
                if feature not in self.feature_names:
                    raise Exception(f"{type(variable)} is missing feature dependency {feature}. Current features: {self.feature_names}")
        self.num_features = len(self.feature_names)
        self.trainable_features = np.array(self.trainable_features)

        self.entry = np.zeros(self.num_features, dtype='float32')

        # Create default shared info and track selector if not provided
        if shared_info is None:
            self.shared_info = SharedInfo()
        else:
            self.shared_info = shared_info
        if track_selector is None:
            self.track_selector = TrackSelector()
        else:
            self.track_selector = track_selector

        # The SharedInfo object contains references to the entry and feature names. This is because some features use information already computed by other features.
        # This is more complicated to implement (you need to define the TrainingVariable.configure()) method, but because there are a number of default variables
        # that depend on other variables (for example the dPhi sums or the RPC compression which is dependent on theta), I made it possible to do this. 
        self.shared_info.feature_names = self.feature_names
        self.shared_info.entry_reference = self.entry

        # Each variable can produce multiple features (for example dPhi can produce dPhi_12, dPhi_13 ect.) Each TrainingVariable declares what features it intends
        # to calculate when it is instantiated. Here, we tell each TrainingVariable where to output what it calculates. 
        start_ind = 0
        for variable in self.variables:
            variable.shared_reference = self.shared_info
            variable.feature_inds = self.entry[start_ind:start_ind + len(variable.feature_names)] # These are the positions in the entry array which this TrainingVariable may populate
            variable.configure()
            start_ind += len(variable.feature_names)

        # These are set by the build_dataset method.
        self.data = None                    # Where all the calculated features are output. There will be a row for each track
        self.event_correspondance = None    # Contains the event number where each track (each row in data) came from 
        self.tracks_processed = 0           # 
        self.events_processed = 0           # 

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
    
    def build_dataset(self, raw_data: dict, tracks_to_process: int) -> np.ndarray:
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

        self.data = np.zeros((tracks_to_process, self.num_features), dtype='float32')
        self.event_correspondance = np.zeros(tracks_to_process, dtype='bool')

        for event_num in range(event_count):
            if event_num % PRINT_EVENT == 0:
                print(f"* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t | Events processed: {event_num}")
                print(f"\t* Trainable tracks: {self.tracks_processed}")
            
            # These ROOT objects work in a strange way. raw_data contains the information for all the events, but when we call GetEntry(i), the branches from entry i become accessible
            for name in tree_names:
                raw_data[name].GetEntry(event_num)

            good_tracks = self.track_selector.select(raw_data)
            
            # Stop if we have already produced enough data
            if self.tracks_processed + len(good_tracks) > tracks_to_process:
                break
            
            for track in good_tracks:
                self.shared_info.calculate(raw_data, track)
                
                self.event_correspondance[self.tracks_processed] = event_num
                self.data[self.tracks_processed] = self.process_event(raw_data)
                self.tracks_processed += 1
        
        self.events_processed = event_num
        self.tracks_processed = self.tracks_processed

        print(f"* Finished Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\t* Events processed: {event_num} \t | Trainable tracks: {self.tracks_processed}")

        return self.data
    
    def get_features(self, features: Union[str, List[str]]) -> np.ndarray:
        """
        Retrieves the specified features from the dataset.

        :param features: A string or a list of feature names to retrieve.
        :return: A numpy array containing the requested features.
        """
        features = np.array([features]) if isinstance(features, str) else np.array(features) 
        for feature in features:
            if feature not in self.feature_names:
                raise Exception(f"{feature} is not a feature in this dataset.")
        
        return np.array(self.data[:, np.isin(np.array(self.feature_names), features)]).squeeze()

    def __str__(self) -> str:
        """
        Returns a string representation of the Dataset object, listing all feature names.

        :return: A string describing the Dataset object with its features.
        """
        return "Dataset object with features: " + ", ".join(self.feature_names)

    def randomize_event_order(self) -> None:
        """
        Randomizes the order of events in the dataset. This affects both the data and the event correspondance.
        
        :return: None
        """
        permutation_inds = np.random.permutation(len(self.data))
        self.data = self.data[permutation_inds]
        self.filtered = self.event_correspondance[permutation_inds]

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

        # When working with reemulated data, the branch names will contain Unp (ex emtfUnpTrack instead of emtfTrack). We must create an alias so that the code can work with these NTuples
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
        :return: A new Dataset object that contains the combined variables and data.
        """
        event_correspondance = datasets[0].event_correspondance

        variables = []
        for dataset in datasets:
            if dataset.event_correspondance != event_correspondance:
                raise Exception("Datasets must have an identical event correspondance")

            variables.extend(dataset.variables)
        
        new_dataset = cls(variables, datasets[0].track_selector, shared_info=datasets[0].shared_info)
        new_dataset.data = np.hstack([dataset.data for dataset in datasets])
        return new_dataset