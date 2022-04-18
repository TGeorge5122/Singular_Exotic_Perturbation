from Option import *
from OptionPricer import OptionPricer
import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    
    locvolCEV = lambda u, t: .2/np.sqrt(u)
    
    rho, nu, kappa = -0.9, 10, 10
    
    paths = 5000
    timesteps = 1000
    
    optionpricer = OptionPricer()
    
    barrier_option = BarrierOption(B = 0.9, K = 1, barrier_type = \
        BarrierType['Down and Out'], option_type = OptionType['Call'])
        
    european_option = EuropeanOption(K = 1, option_type = OptionType['Call'])
    
    #AK = np.arange(.8,1.3,.1)
    AK = np.linspace(0.2, 2.4, 10)
    
    barrier_start_time = time.time()
    
    barrier_prices = optionpricer.localStocVolMC(barrier_option, \
        locvolCEV, S0 = 1, T = 1, AK = AK, rho = rho, nu = nu, kappa = kappa, \
        paths = paths, timesteps = timesteps)
    
    print("--- Barrier option took %s seconds ---" % (time.time() - barrier_start_time))
    
    european_start_time = time.time()    
    
    european_prices = optionpricer.localStocVolMC(european_option, \
        locvolCEV, S0 = 1, T = 1, AK = AK, rho = rho, nu = nu, kappa = kappa, \
        paths = paths, timesteps = timesteps)
        
    print("--- European option took %s seconds ---" % (time.time() - european_start_time))        