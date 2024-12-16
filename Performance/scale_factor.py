import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from Performance.efficiency import get_efficiency

bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
        20,22,24,26,28,30,32,34,36,38,40,42,
        44,46,48,50,60,70,80,90,100,150,200,
        250,300,400,500,600,700,800,900,1000]

def fit_scale_factors(pt, sfa, sfb):
        return sfa / (1 - (sfb * pt))

def get_scale_factor_from_fit(gen_pt, predicted_pt,
                     pt_bins_to_fit_efficiency=bins,
                     pt_cuts_to_fit_sf=np.linspace(1, 30, 30)):

    efficiency, efficiency_err = get_efficiency(gen_pt, predicted_pt, pt_bins_to_fit_efficiency, pt_cut=22)
    efficiency_fit = fit_efficiency(pt_bins_to_fit_efficiency, efficiency, efficiency_err)

    pt_cut = None
    target_efficiency_at_threshold = None
    def target_func(x):
        return theoretical_efficiency(x, pt_cut, *efficiency_fit[1:]) - target_efficiency_at_threshold

    scale_factors = np.zeros(len(pt_cuts_to_fit_sf))
    for i in range(len(pt_cuts_to_fit_sf)):
        target_efficiency_at_threshold = 0.9 * 0.93
        pt_cut = pt_cuts_to_fit_sf[i]
        scale_to = fsolve(target_func, x0=pt_cuts_to_fit_sf[i])[0]
        scale_factors[i] = scale_to / pt_cuts_to_fit_sf[i]

    print(scale_factors)
    bounds = (0, 2)
    scale_factor_fit, pcov = curve_fit(fit_scale_factors, xdata=pt_cuts_to_fit_sf, ydata=scale_factors, p0=[1.237, 0.012])
    return scale_factor_fit, (pt_cuts_to_fit_sf, scale_factors), (pt_bins_to_fit_efficiency, efficiency_fit)

def get_scale_factor_simple(gen_pt, predicted_pt,
                      gen_pt_efficiency_bin_width=4,
                      pt_cuts_to_fit_sf=np.linspace(1, 30, 30)):

    scale_factors = np.zeros(len(pt_cuts_to_fit_sf))
    for i in range(len(pt_cuts_to_fit_sf)):
        # Look at all the muons in a gen pt bin. Find the pt value that 90% of the predicted pts
        # in that bin exceed
        mask = (gen_pt >= pt_cuts_to_fit_sf[i] - gen_pt_efficiency_bin_width / 2) & (gen_pt < pt_cuts_to_fit_sf[i] + gen_pt_efficiency_bin_width / 2)
        scale_factors[i] = pt_cuts_to_fit_sf[i] / np.percentile(predicted_pt[mask], 10)

    scale_factor_fit, pcov = curve_fit(fit_scale_factors, xdata=pt_cuts_to_fit_sf, ydata=scale_factors, p0=[1.237, 0.012])
    return scale_factor_fit, (pt_cuts_to_fit_sf, scale_factors)

def theoretical_efficiency(pt, a, b, c, d):
    return (1/2) * (1 + erf((pt - a) / ((b * (pt ** c) + d ** 2) * np.sqrt(2))))

def fit_efficiency(pt_bins, efficiency, efficiency_err):
    x = pt_bins[:-1] + np.diff(pt_bins) / 2
    popt, pcov = curve_fit(theoretical_efficiency, xdata=x, ydata=efficiency, maxfev=1000000)
    return popt

def current_EMTF_unscale_pt(pt):
    pt_unscale = 1 / (1.07 + 0.015 * pt)
    pt_unscale = np.maximum(pt_unscale, (1 - 0.015 * 20) / 1.07)
    return pt_unscale

def current_EMTF_scale_pt(pt):
    pt_xml = np.minimum(20., pt);  # Maximum scale set by muons with XML pT = 20 GeV (scaled pT ~31 GeV)
    pt_scale = 1.07 / (1 - 0.015 * pt_xml)
    return pt_scale

def generic_scale_pt(pt, a, b, max_pt_to_scale=20):
    pt_xml = np.minimum(max_pt_to_scale, pt);  # Maximum scale set by muons with XML pT = 20 GeV (scaled pT ~31 GeV)
    pt_scale = a / (1 - b * pt_xml)
    return pt_scale