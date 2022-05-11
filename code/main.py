from Option import *
from OptionPricer import OptionPricer
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

def lvSPX(mintau, S0):
    
    a   = 0.0012
    b   =  0.1634
    sig =  0.1029
    rho = -0.5555
    m   = 0.0439
    
    # Local volatility
    
    def lv(S,t):
        k=np.log(S/S0)
        t = np.maximum(t,mintau)
        kp = k/np.sqrt(t)
        v = a + b * (rho * (kp-m) + np.sqrt((kp-m)**2 + sig**2 * t))
        return np.sqrt(abs(v))
        
    return lv

locvolCEV = lambda u, t: .2/np.sqrt(u)

if __name__ == '__main__':
    
    S0 = 1
    T = 1
    AK = np.linspace(0.2, 2.4, 10)
    
    rho, nu, kappa = -0.9, 1, 10
    
    paths = 5000
    timesteps = 1000
    
    locvolMLP = lvSPX(mintau=1/250, S0=S0)
    
    optionpricer = OptionPricer()
    
    barrier_option = BarrierOption(B = 0.9, K = 1, barrier_type = \
        BarrierType['Down and Out'], option_type = OptionType['Call'])
        
    european_option = EuropeanOption(K = 1, option_type = OptionType['Call'])
    
    barrier_start_time = time.time()
    
    barrier_prices = optionpricer.LSVImpact(barrier_option, \
        locvolMLP, S0, T, AK, rho, nu, kappa, \
        paths, timesteps)
    
    print("--- Barrier option took %s seconds ---" % (time.time() - barrier_start_time))
    
    european_start_time = time.time()    
    
    european_prices = optionpricer.LSVImpact(european_option, \
        locvolMLP, S0, T, AK, rho, nu, kappa, \
        paths, timesteps)
        
    print("--- European option took %s seconds ---" % (time.time() - european_start_time))
    
    
    true_lsv_barrier_prices = optionpricer.localStochasticVolMC(barrier_option, \
        locvolMLP, S0, T, AK, rho, nu, kappa, \
        paths, timesteps)