from ExoticOption import *
from ExoticOptionPricer import ExoticOptionPricer
import numpy as np
import time

if __name__ == '__main__':
    
    locvolCEV = lambda u, t: .2/np.sqrt(u)
    
    rho, nu, kappa = 0.9, 0.5, -.95
    
    barrier_option = BarrierOption(B = .7, K = 1, barrier_type = \
        BarrierType['Down and Out'], option_type = OptionType['Call'])
        
    european_option = EuropeanOption(K = 1, option_type = OptionType['Call'])
    
    AK = np.arange(.8,1.3,.1)
    
    barrier_start_time = time.time()
    
    barrier_prices = ExoticOptionPricer.localStocVolMC(barrier_option, \
        locvolCEV, S0 = 1, T = 1, AK = AK, rho = rho, nu = nu, kappa = kappa, \
        paths = 5000, timesteps = 1000)
    
    print("--- Barrier option took %s seconds ---" % (time.time() - barrier_start_time))
    
    european_start_time = time.time()    
    
    european_prices = ExoticOptionPricer.localStocVolMC(european_option, \
        locvolCEV, S0 = 1, T = 1, AK = AK, rho = rho, nu = nu, kappa = kappa, \
        paths = 5000, timesteps = 1000)
        
    print("--- European option took %s seconds ---" % (time.time() - european_start_time))