import numpy as np

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

# STATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions containing station i.
STATION_TRANSITION_MAP = np.array([
    [0, 1, 2], # Example: The first entry here is for station 1. Station 1 appears in the transition list at indices 0, 1, 2 (look at TRANSITION_NAMES)
    [0, 3, 4],
    [1, 3, 5],
    [2, 4, 5]
])

# OUTSTATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions that don't contain station i
OUTSTATION_TRANSITION_MAP = np.array([
    [3, 4, 5], # Example: The first entry here is for station 1. Station 1 does not appear in the transition list at indices 3, 4, 5 (look at TRANSITION_NAMES)
    [1, 2, 5],
    [0, 2, 4],
    [0, 1, 3]
])

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