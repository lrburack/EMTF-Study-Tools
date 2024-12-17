#! /usr/bin/env python

##This file contains the Run-3 training variables for each mode
## the ordering of the variables here does not matter
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