# ML modules:
import xgboost as xgb
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers

import pickle
import config
import os
import numpy as np
from helpers import get_by_name, permute_together
from Dataset import Dataset, Run3TrainingVariables, get_station_presence
import sys

# mode = 15
mode = int(sys.argv[1])
print("MODE: " + str(mode))
# test_train_split = 250000

# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A']
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A']
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A']
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A', "loose_showerCount"]
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A', "loose_showerCount", "nominal_showerCount", "tight_showerCount"]
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A', "careful_shower_bit_thresh=1"]
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi', 'dPhiSum4', 'dPhiSum4A', 'dPhiSum3', 'dPhiSum3A', 'loose_1', 'nominal_1', 'tight_1', 'loose_2', 'nominal_2', 'tight_2', 'loose_3', 'nominal_3', 'tight_3', 'loose_4', 'nominal_4', 'tight_4', 'loose_showerCount', 'nominal_showerCount', 'tight_showerCount', 'shower_bit_thresh=1', 'shower_bit_thresh=2', 'careful_shower_bit_thresh=1']
# use_features = ['theta', 'st1_ring2', 'dPhi_12', 'dPhi_13', 'dPhi_14', 'dPhi_23', 'dPhi_24', 'dPhi_34', 'dTh_14', 'FR_1', 'bend_1', 'RPC_1', 'RPC_2', 'RPC_3', 'RPC_4', 'outStPhi']

use_features = Run3TrainingVariables[str(mode)]
# use_features = None

exclude_features = None
# exclude_features = Run3TrainingVariables[str(mode)]

# study = "FullNtupleStudy/Redone/UnsignedBend/"
# study = "MLStudy/CustomLoss/WeightedMSE/pt_cut=22"
study = "Tests/RateCalculations/Attempt1/"
# study = "ShowerStudy/SectorMatching/Promotion/2Loose_or_1Nominal_LooseShowerBit"
name = "mode=" + str(mode)

training_dataset_name = "Control/mode=" + str(mode)
# testing_dataset_name = "Control/mode=" + str(mode) + "_testing_distribution"
testing_dataset_name = "EphemeralZeroBias/mode=" + str(mode)
# training_dataset_name = "FullNtuple/mode=" + str(mode)
# testing_dataset_name = "FullNtuple/mode=" + str(mode) + "_testing_distribution"
# training_dataset_name = "ShowerDataset/ManyMode/mode=" + str(mode)
# testing_dataset_name = "ShowerDataset/ManyMode/mode=" + str(mode) + "_testing_distribution"
# training_dataset_name = "ShowerDataset/SectorMatching/mode=" + str(mode)
# testing_dataset_name = "ShowerDataset/SectorMatching/mode=" + str(mode) + "_testing_distribution"

if not os.path.exists(os.path.join(config.STUDY_DIRECTORY, study)):
    os.makedirs(os.path.join(config.STUDY_DIRECTORY, study))

# ----------------------------------- Train -----------------------------------
dataset = get_by_name(training_dataset_name, config.WRAPPER_DICT_NAME)['training_data_builder']

if use_features == None:
    use_features = np.array(dataset.feature_names)[dataset.trainable_features]

if exclude_features != None:
    use_features = [feature for feature in use_features if feature not in exclude_features]

# unset_mask = dataset.get_features("emtfHit_slope_1", filtered=False) == -99
# dataset.filtered = np.logical_and(dataset.filtered, np.logical_not(unset_mask))

training_data = dataset.get_features(use_features)          #[:test_train_split]
train_pt = np.log(dataset.get_features(["gen_pt"]))         #[:test_train_split]
event_weight = 1 / np.log2(dataset.get_features(["gen_pt"]))#[:test_train_split]

training_data = dataset.get_features(use_features)

print("* Training on " + str(len(training_data)) + " events")

# # Define a neural network model
# model = models.Sequential()
# model.add(layers.Input(shape=(training_data.shape[1],)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='linear'))  # Output layer

# # Compile the model
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
#               loss='mse',  # Mean Squared Error Loss
#               metrics=['mae'])  # Mean Absolute Error

# # Train the model
# model.fit(training_data, train_pt, sample_weight=event_weight.flatten(),
#           epochs=50, batch_size=1024, verbose=1)  # Adjust epochs and batch_size as needed

# # Save the trained model
# model_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_trained_NN" + '.h5')
# model.save(model_path)

# print("------------------------------ Model Summary ---------------------------")
# model.summary()

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', 
                        learning_rate = .1, 
                        max_depth = 5, 
                        n_estimators = 400,
                        nthread = 30)

xg_reg.fit(training_data, train_pt, sample_weight = event_weight)

model_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.MODEL_NAME)
with open(model_path, 'wb') as file:
    pickle.dump(xg_reg, file)

feature_importances_str = ""

feature_importances_str += "------------------------------ Feature Importances ---------------------------\n"
feature_importances = xg_reg.feature_importances_
for feature_name, importance in zip(use_features, feature_importances):
    feature_importances_str += f"{feature_name}:\t {importance}" + "\n"
feature_importances_str += "\n------------------------------------------------------------------------------"
print(feature_importances_str)

feature_importances_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + "feature_importances.txt")
f = open(feature_importances_path, "a")
f.write(feature_importances_str)
f.close()

# ----------------------------------- Predict -----------------------------------
dataset = get_by_name(testing_dataset_name, config.WRAPPER_DICT_NAME)['training_data_builder']
testing_data = dataset.get_features(use_features)[:202888]

print("* Testing on " + str(len(testing_data)) + " events")

predicted_pt = np.exp(xg_reg.predict(testing_data))

# features = ["loose_" + str(station + 1) for station in np.where(get_station_presence(mode))[0]]
# promote = np.sum(dataset.get_features(features), axis=1).flatten() >= 2
# features = ["nominal_" + str(station + 1) for station in np.where(get_station_presence(mode))[0]]
# promote2 = np.sum(dataset.get_features(features), axis=1).flatten() >= 1

# promote = np.logical_or(promote, promote2)

# predicted_pt[promote] = 200

print(np.sum(predicted_pt > 15.7))

# test_dict = {
#     "training_dataset"   : training_dataset_name,
#     "testing_dataset"   : testing_dataset_name,
#     "predicted_pt"      : predicted_pt,
#     "training_features" : use_features,
#     "gen_features"      : np.array(dataset.feature_names)[~dataset.trainable_features],
#     # "gen_data"          : dataset.get_features(["gen_pt", "gen_eta", "gen_phi"])
# }

# prediction_path = os.path.join(config.STUDY_DIRECTORY, study, name + "_" + config.PREDICTION_NAME)
# print(prediction_path)
# with open(prediction_path, 'wb') as file:
#     pickle.dump(test_dict, file)