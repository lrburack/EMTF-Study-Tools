# Change this file for your local clone of the repository
CODE_DIRECTORY      = "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF_BDT"            # Change this!
DATASET_DIRECTORY   = "/eos/user/l/lburack/work/BDT_studies/Results/Datasets"       # And this!
STUDY_DIRECTORY     = "/eos/user/l/lburack/work/BDT_studies/Results/Studies"
FIGURE_DIRECTORY   = "/afs/cern.ch/user/l/lburack/work/BDTdev/Figures"              # And this!

WRAPPER_DICT_NAME = "wrapper_dict.pkl"
MODEL_NAME = "xgb_model.pkl"
PREDICTION_NAME = "prediction.pkl"


# Put the paths of the available ntuples here with a sensible name.
# You may need to request access to these folders.

# Training distribution (flat in 1/pT)
wHMT = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240826_193940/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240826_193901/0000"]
# Testing distribution (flat in pT)
wHMT_testing_distribution = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/241008_145954/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/241008_145818/0000"]

# With the very loose
# Training distribution (flat in 1/pT)
wHMT_VeryLoose = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240924_141836/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240924_142146/0000"]
# Testing distribution (flat in pT)
wHMT_VeryLoose_testing_distribution = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240925_163603/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240925_163514/0000"]

# ZMuSkim (used for getting realistic efficiency plots)
ZMuSkim = ["/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/Muon0/Muon_ZmuSkim_14_0_17_rawReco_wCscSegmentsShowers_v2/240926_131840/0000/"]

# EphemeralZeroBias (used for calculating rates)
EphemeralZeroBias = ["/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/EphemeralZeroBias0/EphemeralZeroBias0_Run2024I_v2/241105_153204/0000/"]
