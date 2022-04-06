'''
This code implemments the exotic options pricer.
The exotic options price can be compute under local volaility or local stochastic
volatility.
The user provides their own local volatility function and pricing under
LSV is computed using the LSV impact formula presented in Monicaud and Reghai.
'''

from BlackScholes import BSFormula, BSImpliedVolCall
from ExoticOption import ExoticOption
from typing import Callable
import numpy as np
import pandas as pd

class ExoticOptionPricer:   
    
    #Computes the value of the exotic option under the passed local volatility
    #function using monte carlo simulations.
    @staticmethod
    def localVolMC(exoticoption: ExoticOption, locvol: Callable[[float, float], float], S0, T, AK, paths, timesteps, sigma_shift = 0):
        
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
            sig = locvol(S0*np.exp(s[i]),t/2) + sigma_shift
            s[i+1] = s[i] - sig**2/2*dt + sig*sqdt * W
            t += dt
            
            
        option_value = pd.DataFrame(columns = ['Strike','Value under Local Volatility'])
        
        S = S0*np.exp(s)
        # Evaluate mean call value for each path
        
        for K in AK:
            
            exoticoption.set_strike(K)            
            option_value = option_value.append({'Strike': K, 'Value under Local Volatility': \
                                                np.nanmean(exoticoption.payoff(S))}, ignore_index=True)
            
        return option_value
        
    #Computes the value of the exotic option under local stochastic volatility
    #using the passed local volatility function and the LSV impact.
    @staticmethod    
    def localStocVolMC(exoticoption: ExoticOption, locvol: Callable[[float, float, float], float], S0, T, AK, rho, nu, kappa, paths, timesteps):
        
        delta_S = 1e-6
        delta_sigma = 1e-6
        
        #Value of the options under local volatility model
        option_value = ExoticOptionPricer.localVolMC(exoticoption, locvol, S0, T, AK, paths, timesteps)
        
        #Compute the LSV Impact
        sigma_y_2 = nu ** 2 / (2 * kappa)
        
        sigma_kt = 
        
        #Well choosen scenario such that x_kt_volga makes the exotic vanna disappear for vanilla options
        x_kt_vanna = 
        
        #Well choosen scenario such that x_kt_volga makes the exotic volga disappear for vanilla options
        x_kt_volga = 
        
        exotic_vanna = ExoticOptionPricer.localVolMC(exoticoption, locvol, S0 + delta_S, T, AK, paths, timesteps, delta_sigma)['Value under Local Volatility'] \
            - ExoticOptionPricer.localVolMC(exoticoption, locvol, S0 + delta_S, T, AK, paths, timesteps)['Value under Local Volatility'] \
            - ExoticOptionPricer.localVolMC(exoticoption, locvol, S0, T, AK, paths, timesteps, x_kt_vanna)['Value under Local Volatility'] \
            + option_value['Value under Local Volatility']
        
        exotic_volga = ExoticOptionPricer.localVolMC(exoticoption, locvol, S0, T, AK, paths, timesteps, -delta_sigma)['Value under Local Volatility'] \
            - 2 * option_value['Value under Local Volatility'] \
            + ExoticOptionPricer.localVolMC(exoticoption, locvol, S0, T, AK, paths, timesteps, x_kt_volga)['Value under Local Volatility']
        
        
        option_value['LSV Impact'] = rho * nu / kappa * exotic_vanna + 1 / 2 * sigma_y_2 * exotic_volga
        
        option_value['Value under Local Stochastic Volatility'] = option_value['LSV Impact'] + option_value['Value under Local Volatility']
        
        return option_value