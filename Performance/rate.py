from scipy.integrate import quad
import numpy as np

def get_rate_zerobias(pt, nevents, pt_cut):
    return (np.sum(pt > pt_cut) / nevents) * 40e9

def get_rate(efficiency, bins, pt_cut):
    a = 9.46308e+05
    b = 0.76018
    c = -3.66
    def muon_frequency(pt):
        A = 9.463e5
        b = -0.76018
        c = -3.66

        return A * ((pt - b) ** c)
    
    passing_muon_count = np.zeros(len(efficiency))

    for i in range(len(bins) - 1):
        passing_muon_count[i], _ = quad(muon_frequency, bins[i], bins[i+1])
    
    return passing_muon_count * efficiency

