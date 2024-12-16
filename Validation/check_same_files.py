import os
import numpy as np
import pickle
import sys
sys.path.insert(0, "/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools")
import config

paths_to_check = np.array([
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/AllGenMuonsFromTestingDistribution/allgenmuons/exclude_pos79",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=15",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=14",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=13",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=11",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=10",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=7",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=6",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=5",
    "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/Rates/mode=3",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/ShowerStudy/MatchChamber/mode=15_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/ShowerStudy/MatchSector/mode=13_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/mode=15_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/mode=14_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/mode=10_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/mode=3_testing_distribution",
    # "/eos/home-l/lburack/work/BDT_studies/Results/Datasets/BDT2025/Control/mode=13_testing_distribution",
])

print(paths_to_check)

first_files = None

for path in paths_to_check:

    print(f"{path}...")

    files = []

    with open(os.path.join(path, config.WRAPPER_DICT_NAME), "rb") as file:
        a = pickle.load(file)
        
    if os.path.exists(os.path.join(path, "build_dataset.out")):
        with open(os.path.join(path, "build_dataset.out"), "r") as file:
            for line in file:
                if line.startswith("* Loading file"):
                    files.append(line.split(": ")[-1][:-1])
    else:
        # print(a.keys())
        files = a["file_names"]
    
    files = np.array(files)

    if first_files is None:
        first_files = files
        print(f"Found {len(files)} files")
        continue

    if np.all(np.isin(files, first_files)) and len(files) == len(first_files):
        print("\tidentical!")
    else:
        print(f"Not matching. {len(files)} files")
        print(f"Missing: {first_files[~np.isin(first_files, files)]}")
        print(f"Extra: {files[~np.isin(files, first_files)]}")

    # if isinstance(a, dict):
    #     if "file_names" not in list(a.keys()):
    #         print("File names attribute did not exist. Resaving dict")
    #         a["file_names"] = files
    #         # with open(os.path.join(path, config.WRAPPER_DICT_NAME), "wb") as file:
    #         #     pickle.dump(a, file)
    # else:
    #     print(f"overwritten you idiot. it had {len(files)} files")