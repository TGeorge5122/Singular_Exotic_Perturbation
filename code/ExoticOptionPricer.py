'''
This code implemments the exotic options pricer.
The exotic options price can be compute under local volaility or local stochastic
volatility.
The user provides their own local volatility function and pricing under
LSV is computed using the LSV impact formula presented in Monicaud and Reghai.
'''

from ExoticOption import ExoticOption
from typing import Callable
import numpy as np
import pandas as pd

class ExoticOptionPricer:
    
    @staticmethod
    def localVolMC(exoticoption: ExoticOption, locvol: Callable[[float, float], float], S0, T, AK, paths, timesteps):
        
        dt = T / timesteps
        sqdt = np.sqrt(dt)
      
        # We use a vertical array, one element per M.C. path
        s = np.zeros((timesteps + 1, paths))
        t = np.zeros(paths)
        
        for i in range(timesteps):
            
            W = np.random.normal(size = paths)
            W -= np.mean(W)
            W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
            
            # Stock SDE discretization
            sig = locvol(S0*np.exp(s[i]),t/2)
            s[i+1] = s[i] - sig**2/2*dt + sig*sqdt * W
            t += dt
            
            
        results = pd.DataFrame(columns = ['AK','AV'])
        
        S = S0*np.exp(s)
        # Evaluate mean call value for each path
        
        for K in AK:
            
            exoticoption.set_strike(K)            
            results = results.append({'AK': K, 'AV': np.nanmean(exoticoption.payoff(S))}, ignore_index=True)
            
        return results
    
    @staticmethod    
    def localStocVolMC():
        pass