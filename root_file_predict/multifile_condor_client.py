import sys
import os

# NOTE. Because I put this in a subdirectory, unfortunately you will have to manually add this path here, AND in root_file_predict.py.
sys.path.append("/afs/cern.ch/user/l/lburack/work/BDTdev/EMTF-Study-Tools")

import config
from helpers import path_to_xrd


# Modify root_file_predict to your liking before running this script.
# This script will submit a condor job to run root_file_predict on each file in the input_base_path

input_base_paths = {'endcap1': config.BDT2025[0], 'endcap2': config.BDT2025[1]}
files_per_endcap = 1

output_base_path = "/eos/user/l/lburack/work/BDT_studies/Results/Studies/TestRootFilePredict/test2"
condor_sub_path = os.path.join(config.CODE_DIRECTORY, "root_file_predict/root_file_predict.sub")

# Prepare output directories
os.makedirs(output_base_path, exist_ok=True)
for endcap in input_base_paths.keys():
    os.makedirs(os.path.join(output_base_path, endcap), exist_ok=True)

# Gather files
files = []

for endcap, base_path in input_base_paths.items():
    collected_files = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        rel_dir = os.path.relpath(dirpath, base_path)
        output_dir = os.path.join(output_base_path, endcap, rel_dir)
        os.makedirs(output_dir, exist_ok=True)

        for file in filenames:
            if file.endswith(".root"):
                rel_file_path = os.path.join(endcap, rel_dir, file)
                collected_files.append(rel_file_path)

                if len(collected_files) >= files_per_endcap:
                    break

        if len(collected_files) >= files_per_endcap:
            break

    files.extend(collected_files)

# Submit condor jobs
for file in files:
    endcap = file.split('/')[0]
    input_dir = os.path.join(input_base_paths[endcap], file[len(endcap) + 1:])

    output_dir = path_to_xrd(os.path.join(output_base_path, file))
    unique_name = file.replace("/", "_").replace(".root", "")  # This is just used for the log file.

    command = f"condor_submit {condor_sub_path} code_directory={config.CODE_DIRECTORY} input_dir={input_dir} output_uri={output_dir} unique_name={unique_name}"
    os.system(command)