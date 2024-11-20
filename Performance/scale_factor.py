
def get_scale_factor(gen_pt, predicted_pt, pt_cut):
    bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
        20,22,24,26,28,30,32,34,36,38,40,42,
        44,46,48,50,60,70,80,90,100,150,200,
        250,300,400,500,600,700,800,900,1000]
    efficiency, efficiency_err = get_efficiency(gen_pt, predicted_pt, bins, pt_cut)
    popt = fit_efficiency(bins, efficiency, efficiency_err)
    
    def target_func(x):
        return theoretical_efficiency(x, *popt) - 0.9
    
    pt_90 = fsolve(target_func, pt_cut)[0]
    return pt_90 / pt_cut