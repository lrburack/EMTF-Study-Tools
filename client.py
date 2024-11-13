import pickle
import os
import argparse
import config
from helpers import build_from_wrapper_dict

from Dataset import *

from Objects.Default.Variables import *
from Objects.Default.SharedInfo import *
from Objects.Default.TrackSelectors import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--condor", required=False, default=0)
args = parser.parse_args()

CONDOR = bool(args.condor)

# You may need to request access to these folders
# Training distribution (flat in 1/pT)
base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240826_193940/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240826_193901/0000"]
# Testing distribution (flat in pT)
# base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/241008_145954/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/241008_145818/0000"]

# With the very loose
# Training distribution (flat in 1/pT)
# base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240924_141836/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240924_142146/0000"]
# Testing distribution (flat in pT)
# base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240925_163603/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240925_163514/0000"]

# ZMuSkim
# base_dirs = ["/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/Muon0/Muon_ZmuSkim_14_0_17_rawReco_wCscSegmentsShowers_v2/240926_131840/0000/"]

# EphemeralZeroBias
# base_dirs = ["/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/EphemeralZeroBias0/EphemeralZeroBias0_Run2024I_v2/241105_153204/0000/"]

mode = 15

# name = "Control/mode=" + str(mode) + "_testing_distribution"
# name = "ShowerDataset/SectorMatching/mode=" + str(mode)
# name = "ShowerDataset/SectorMatching/mode=" + str(mode) + "_testing_distribution"
# name = "FullNtuple/mode=" + str(mode)
# name = "FullNtuple/mode=" + str(mode) + "_testing_distribution"
# name = "EphemeralZeroBias/mode=" + str(mode)
name = "Tests/mode=" + str(mode)

# Make sure you don't accidently overwrite an existing dataset
if os.path.exists(os.path.join(config.RESULTS_DIRECTORY, name)) and os.path.isdir(os.path.join(config.RESULTS_DIRECTORY, name)):
    print("A dateset with the name " + name + " has already been initiated.")
    overwrite = ""
    while overwrite not in ["y", "n"]:
        overwrite = input("Overwrite it (y/n)? ").lower()
    
    if overwrite == "n":
        exit()

# track_branches = ["emtfTrack_dxy", "emtfTrack_phi", "emtfTrack_phi_fp", "emtfTrack_eta", "emtfTrack_q", "emtfTrack_mode", "emtfTrack_endcap", "emtfTrack_sector", "emtfTrack_bx", "emtfTrack_nhits"]
# hit_branches = ["emtfHit_endcap", "emtfHit_station", "emtfHit_ring", "emtfHit_sector", "emtfHit_subsector", "emtfHit_chamber", "emtfHit_cscid", "emtfHit_bx", "emtfHit_type", "emtfHit_neighbor", "emtfHit_strip", "emtfHit_strip_qses", "emtfHit_wire", "emtfHit_roll", "emtfHit_layer", "emtfHit_quality", "emtfHit_pattern", "emtfHit_bend", "emtfHit_slope", "emtfHit_time", "emtfHit_emtf_phi", "emtfHit_emtf_theta"]

training_data_builder = Dataset(variables=[
                                           GeneratorVariables.for_mode(mode), 
                                        #    RecoVariables.for_mode(mode), 
                                           Theta.for_mode(mode),
                                           St1_Ring2.for_mode(mode),
                                           dPhi.for_mode(mode),
                                           dTh.for_mode(mode),
                                           FR.for_mode(mode),
                                           RPC.for_mode(mode),
                                           Bend.for_mode(mode),
                                           OutStPhi.for_mode(mode),
                                           dPhiSum4.for_mode(mode),
                                           dPhiSum4A.for_mode(mode),
                                           dPhiSum3.for_mode(mode),
                                           dPhiSum3A.for_mode(mode),
                                        #    TrackVariables(track_branches),
                                        #    HitVariables(hit_branches, stations=stations(mode))
                                        #    AllShower(stations(mode), match_by="sector"),
                                        #    ShowerCount(stations(mode), ["loose", "nominal", "tight"]),
                                        #    ShowerStationType(stations(mode)),
                                        #    CarefulShowerBit(stations(mode), shower_threshold=1),
                                        #    CarefulShowerBit(stations(mode), shower_threshold=2),
                                        #    CarefulShowerBit(stations(mode), shower_threshold=3)
                                           ],
                                track_selector=TrackSelector(15),
                                shared_info=SharedInfo(mode=mode),
                                # compress=True
                                )


wrapper_dict = {
    'training_data_builder': training_data_builder,
    'base_dirs': base_dirs,
    'files_per_endcap': 1,
    'tracks_to_process': 500
}

os.makedirs(os.path.join(config.RESULTS_DIRECTORY, name), exist_ok=True)
dict_path = os.path.join(config.RESULTS_DIRECTORY, name, config.WRAPPER_DICT_NAME)

if CONDOR:
    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)
    
    condor_submit_path = os.path.join(config.CODE_DIRECTORY, "condor_wrapper.sub")
    command = "condor_submit " + condor_submit_path + " code_directory=" + config.CODE_DIRECTORY + " results_directory=" + config.RESULTS_DIRECTORY + " name=" + name

    print(command)
    os.system(command)
else:
    # Builds the dataset in-place
    build_from_wrapper_dict(wrapper_dict)

    print(dict_path)
    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)

    # print("Dir contents: " + str(os.listdir(dict_path)))