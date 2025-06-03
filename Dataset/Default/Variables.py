from Dataset.Dataset import TrainingVariable
from Dataset.constants import *

# -------------------------------    Training variable definitions    -----------------------------------
class GeneratorVariables(TrainingVariable):
    def __init__(self):
        super().__init__(["gen_pt", "gen_eta", "gen_phi", "gen_q"], tree_sources=["EMTFNtuple"], trainable=False)
    
    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].genPart_pt[0]
        self.feature_inds[1] = event['EMTFNtuple'].genPart_eta[0]
        self.feature_inds[2] = event['EMTFNtuple'].genPart_phi[0]
        self.feature_inds[3] = event['EMTFNtuple'].genPart_q[0]

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
        signs = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)])
        deltaPh = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_deltaPh[int(self.shared_reference.track)])
        signs = np.where(signs[self.transition_inds], 1, -1)
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
        elif mode == 14 or mode == 13 or mode == 11:
            # If the first RPC is present we dont care about the others
            if rpc[0]:
                self.feature_inds[1] = 0
                self.feature_inds[2] = 0
            elif rpc[2]:
                self.feature_inds[1] = 0
        # No RPC compression for 2 station modes (these bits are not included in the LUT address)
        # We just do two 3 bit bends for the first two present stations

    @classmethod
    def for_mode(cls, mode):
        # There should be an RPC feature for each station in the track
        return cls(stations(mode))


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


# Added for the root_file_predict.py script to know where to put the predicted pt in the root file
class Track(TrainingVariable):
    def __init__(self):
        super().__init__(["track"], tree_sources=["EMTFNtuple"], trainable=False)

    def calculate(self, event):
        self.feature_inds[0] = int(self.shared_reference.track)

    @classmethod
    def for_mode(cls, mode):
        return cls()