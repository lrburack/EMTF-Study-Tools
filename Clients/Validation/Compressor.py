dPhiNLBMap_4bit_256Max = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 31, 46, 68, 136]
dPhiNLBMap_5bit_256Max = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136]
dPhiNLBMap_7bit_512Max = [
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  79,  80,  81,  83,  84,  86,  87,  89,  91,  92,  94,
    96,  98,  100, 102, 105, 107, 110, 112, 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168,
    174, 181, 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470]

def getNLBdPhi(dPhi, bits, max):
        dPhi_ = max
        sign_ = 1
        if dPhi < 0:
            sign_ = -1
        dPhi = sign_ * dPhi

        if max == 256:
            if bits == 4:
                dPhi_ = dPhiNLBMap_4bit_256Max[(1 << bits) - 1]
                for edge in range ((1 << bits) - 1):
                    if dPhiNLBMap_4bit_256Max[edge] <= dPhi and dPhiNLBMap_4bit_256Max[edge + 1] > dPhi:
                        dPhi_= dPhiNLBMap_4bit_256Max[edge]
            if bits == 5:
                dPhi_ = dPhiNLBMap_5bit_256Max[(1 << bits) - 1]
                for edge in range((1 << bits) - 1):
                    if dPhiNLBMap_5bit_256Max[edge] <= dPhi and dPhiNLBMap_5bit_256Max[edge + 1] > dPhi:
                        dPhi_ = dPhiNLBMap_5bit_256Max[edge]
        elif max == 512:
            if bits == 7:
                dPhi_ = dPhiNLBMap_7bit_512Max[(1 << bits) - 1]
                for edge in range((1 << bits) - 1):
                    if dPhiNLBMap_7bit_512Max[edge] <= dPhi and dPhiNLBMap_7bit_512Max[edge + 1] > dPhi:
                        dPhi_ = dPhiNLBMap_7bit_512Max[edge]


        return sign_ * dPhi_



class Compressor(dict):
    def __init__(self):
        super(Compressor, self).__init__()
        
    def compress(self):
        theta = self['theta']
        mode = int(self['mode'])
        if mode == 15:
            if self["st1_ring2"] == 0:
                theta = (min(max(theta, 5), 52) - 5) / 6
            else:
                theta = ((min(max(theta, 46), 87) - 46) / 7) + 8
        else: 
            if not self["st1_ring2"]: theta = (max(theta, 1) - 1) / 4
            else: theta = ((min(theta, 104) - 1) / 4) + 6
        self['theta'] = int(theta)
        
        if mode == 15 and not self["st1_ring2"]:
            if theta < 4:
                self['RPC_3'] = 0
                self['RPC_4'] = 0

        nRPCs = sum(self['RPC_' + str(i + 1)] if self['presence_' + str(i+1)] else 0 for i in range(4))

        if nRPCs >= 2:
            if mode == 15:
                if (self['RPC_1'] and self['RPC_2']):
                    self['RPC_3'] = 0
                    self['RPC_4'] = 0
                elif (self['RPC_1'] and self['RPC_3']):
                    self['RPC_4'] = 0
                elif (self['RPC_4'] and self['RPC_2']):
                    self['RPC_3'] = 0
                elif (self['RPC_3'] and self['RPC_4'] and not self['st1_ring2']):
                    self['RPC_3'] = 0
            elif mode == 14:
                if self['RPC_1']:
                    self['RPC_2'] = 0
                    self['RPC_3'] = 0
                elif self['RPC_3']:
                    self['RPC_2'] = 0
            elif mode == 13:
                if self['RPC_1']:
                    self['RPC_2'] = 0
                    self['RPC_4'] = 0
                elif self['RPC_4']:
                    self['RPC_2'] = 0
            elif mode == 11:
                if self['RPC_1']:
                    self['RPC_3'] = 0
                    self['RPC_4'] = 0
                elif self['RPC_4']:
                    self['RPC_3'] = 0
            elif mode == 7:
                if self['RPC_2']:
                    self['RPC_3'] = 0
                    self['RPC_4'] = 0
                elif self['RPC_4']:
                    self['RPC_3'] = 0



                    
        
        
        
        for i in ["12", "13", "14", "23", "24", "34"]:
            dTheta = self["dTh_" + i]
            if dTheta == -999: continue
            if mode == 15:
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
            
            self["dTh_" + i] = dTheta

            dPhi = self["dPhi_" + i]
            
            nBitsA = 7
            nBitsB = 7
            nBitsC = 7
            maxA = 512
            maxB = 512
            maxC = 512

            if mode == 7 or mode == 11 or mode > 12:
                nBitsB = 5
                maxB = 256
                nBitsC = 5
                maxC = 256
            if mode == 15:
                nBitsC = 4
                maxC = 256


            
            if i == '23':
                if mode == 7:
                    dPhi = getNLBdPhi(dPhi, nBitsA, maxA)
                else:
                    dPhi = getNLBdPhi(dPhi, nBitsB, maxB)
            elif i == '24':
                dPhi = getNLBdPhi(dPhi, nBitsB, maxB)
            elif i == '34':
                dPhi = getNLBdPhi(dPhi, nBitsC, maxC)
            else:
                dPhi = getNLBdPhi(dPhi, nBitsA, maxA)    

            self['dPhi_' + i] = dPhi
        
        if mode == 15:
            self['dPhi_13'] = self['dPhi_12'] + self['dPhi_23']
            self['dPhi_14'] = self['dPhi_13'] + self['dPhi_34']
            self['dPhi_24'] = self['dPhi_23'] + self['dPhi_34']
        elif mode == 14:
            self['dPhi_13'] = self['dPhi_12'] + self['dPhi_23']
        elif mode == 13:
            self['dPhi_14'] = self['dPhi_12'] + self['dPhi_24']
        elif mode == 11:
            self['dPhi_14'] = self['dPhi_13'] + self['dPhi_34']
        elif mode == 7:
            self['dPhi_24'] = self['dPhi_23'] + self['dPhi_34']

        dPhSign = self['signPhi']
        nBits = 2 if mode == 7 or mode == 11 or mode > 12 else 3
        for i in range(4):
            endcap = self['endcap']
            sign_ = endcap * -1 * int(dPhSign)
            clct_ = 0

            pattern = self['pattern_' + str(i + 1)]
            if self['presence_' + str(i + 1)]:
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
                    elif pattern == 0 and not self['RPC_' + str(i + 1)]:
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
            self['bend_' + str(i + 1)] = clct_