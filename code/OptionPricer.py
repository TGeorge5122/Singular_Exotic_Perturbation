'''
This code implemments the exotic options pricer.
The exotic options price can be compute under local volaility or local stochastic
volatility.
The user provides their own local volatility function and pricing under
LSV is computed using the LSV impact formula presented in Monicaud and Reghai.
'''

from BlackScholes import BlackScholes
from Option import Option, EuropeanOption, OptionType
from typing import Callable
from scipy import interpolate
import numpy as np
import pandas as pd
import pdb

class OptionPricer:
    
    #default constructor
    def __init__(self):
        self.x_volga_kt = None
        self.x_vanna_kt = None
        self.delta_sigma = 0.03
        self.delta_S = 0.03
        self.beta = 0.03
    
    #Computes the value of the exotic option under the passed local volatility function using mc simulations
    @staticmethod
    def localVolMC(option: Option, locvol: Callable[[float, float], float], S0, T, AK, paths, timesteps, deformation=0, debug = False):
        
        dt = T / timesteps
        sqdt = np.sqrt(dt)
        deformation=np.exp(deformation)
        
        # We use a vertical array, one element per M.C. path
        s = np.zeros((timesteps + 1, paths))
        t = np.zeros(paths)
        
        for i in range(timesteps):
            
            if debug:
                pdb.set_trace()        
            
            W = np.random.normal(size = paths)
            W -= np.mean(W)
            W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
            
            # Stock SDE discretization
            sig = locvol(S0*np.exp(s[i]), t) * deformation
            s[i+1] = s[i] - sig**2/2*dt + sig*sqdt * W
            t += dt
        
        S = S0*np.exp(s)
        
        option_value = pd.DataFrame(columns = ['Strike','Value under Local Volatility','Black Scholes Implied Vol'])

        # Evaluate mean call value for each path
        for K in AK:
                        
            value = np.nanmean(option.payoff(S, K))
            bs_vol = BlackScholes.BSImpliedVol(S0, K, T, 0, value, OptionType['Call'])
            
            option_value = option_value.append({'Strike': K, 'Value under Local Volatility': \
                                                value, 'Black Scholes Implied Vol': bs_vol}, ignore_index=True)
                
        return option_value
    
    #Computes the value of the exotic option under the passed local stochastic volatility funtion using mc simulations
    @staticmethod
    def localStochasticVolMC(option: Option, locvol: Callable[[float, float], float], S0, T,  AK, rho, nu, kappa, paths, timeSteps, epsilon = 1, debug = False):
    
        dt = T / timeSteps
        sqdt = np.sqrt(dt)
        
        #Ornstein-Uhlenbeck process Y(t) coefficients
        theta   = kappa / epsilon
        sigma   = nu / np.sqrt(epsilon)
        sigma_2 = np.square(nu / np.sqrt(epsilon))
        
        #Y_t is normally distributed so E[e^2Y] is normal mgf with t = 2
        #Variance of Y(t) = sigma^2 / (2 * theta) (1 - e^(-2 * theta * t))
        f_adjustment = lambda t: np.sqrt(np.exp(sigma_2 / theta * (1 - np.exp(-2 * theta * t))))
        
        # We use a vertical array, one element per M.C. path
        s, Y_t = np.zeros((timeSteps + 1, paths)), np.zeros(paths)
        t = 0
    
        for i in range(timeSteps):
            
            if debug:
                pdb.set_trace()
            
            #Brownian motion for dS(t)
            W = np.random.normal(size = paths)
            W -= np.mean(W)
            W /= np.std(W,ddof=1)
            
            #Used to created correlated Brownian motion for dY(t)
            W_2 = np.random.normal(size = paths)
            W_2 -= np.mean(W)
            W_2 /= np.std(W,ddof=1)
            
            B = rho * W + np.sqrt(1 - rho ** 2) * W_2
            
            sig = locvol(S0*np.exp(s[i]), t) * np.exp(Y_t) / f_adjustment(t)
            
            Y_t = np.exp(-theta * dt) * Y_t + sigma * sqdt * B  
            
            s[i+1] = s[i] - sig**2 / 2 * dt + sig * sqdt * W
            
            t += dt
        
        S = S0*np.exp(s)
        
        option_value = pd.DataFrame(columns = ['Strike','Value under Local Stochastic Volatility'])

        # Evaluate mean call value for each path
        for K in AK:
                        
            option_value = option_value.append({'Strike': K, 'Value under Local Stochastic Volatility': \
                                                np.nanmean(option.payoff(S, K))}, ignore_index=True)
    
        return option_value
    
    #Computes the value of the exotic option under local stochastic volatility
    #using the passed local volatility function and the LSV impact.
    def LSVImpact(self, option: Option, locvol: Callable[[float, float, float], float], S0, T, AK, rho, nu, kappa, paths, timesteps, r = 0, deformation=0,debug = False):
        
        if not self.x_vanna_kt or not self.x_volga_kt:
            self.xKT(option, locvol, S0, T, r, paths, timesteps)
            
        p_kt_sigma_x_vanna = self.localVolMC(option, lambda u, t: locvol(u,t) + self.x_vanna_kt(u,t,grid=False).flatten(), S0, T, AK, paths, timesteps)
        p_kt_sigma_x_volga = self.localVolMC(option, lambda u, t: locvol(u,t) + self.x_volga_kt(u,t,grid=False).flatten(), S0, T, AK, paths, timesteps) 
        p_kt_sigma = self.localVolMC(option, locvol, S0, T, AK, paths, timesteps)
        
        #p_kt_s_sigma_perturb = self.localVolMC(option, lambda u, t: locvol(u,t) + self.delta_sigma, S0 + self.delta_S, T, AK, paths, timesteps)
        p_kt_s_sigma_perturb = self.localVolMC(option, lambda u, t: locvol(u,t), S0 + self.delta_S, T, AK, paths, timesteps, deformation = self.beta)
        
        #p_kt_sigmaperturb = self.localVolMC(option, lambda u, t: locvol(u,t) - self.delta_sigma, S0, T, AK, paths, timesteps)
        p_kt_s_perturb = self.localVolMC(option, locvol, S0 + self.delta_S, T, AK,  paths, timesteps)
        p_kt_negative_beta = self.localVolMC(option, locvol, S0 , T, AK, paths, timesteps, deformation = -self.beta)
        
        exotic_volga = (p_kt_sigma_x_volga['Value under Local Volatility'].values - (2 * p_kt_sigma['Value under Local Volatility'] - p_kt_negative_beta['Value under Local Volatility']).values) / self.beta**2
        exotic_vanna = (p_kt_s_sigma_perturb['Value under Local Volatility'].values - p_kt_s_perturb['Value under Local Volatility'].values + p_kt_sigma['Value under Local Volatility'].values - p_kt_sigma_x_vanna['Value under Local Volatility'].values) / (self.delta_S * self.delta_sigma) * S0
        
        lsv_impact = 1 / 2 * (nu ** 2 / (2 * kappa)) * exotic_volga + rho * nu / kappa * exotic_vanna
        
        return pd.DataFrame([AK, p_kt_sigma['Value under Local Volatility'].values, lsv_impact, p_kt_sigma['Value under Local Volatility'].values + lsv_impact, exotic_vanna, exotic_volga], \
                            index = ['Strike','Local Volatility','LSV Impact','Local Stochastic Volatility', 'exotic_vanna', 'exotic_volga']).T
    
    def xKT(self, option: Option, locvol: Callable[[float, float, float], float], S0, T, r, paths, timesteps):
        
        print('Computing x_kt')
        
        vanillaoption = EuropeanOption(option.K, OptionType['Call'])
        
        k_array = np.linspace(0.2, 2.4, 10)
        t_array = np.linspace(0, 1, len(k_array))
        t_array[0] = 1/250
        
        volga_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
        vanna_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
        
        sigma_kt_x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
        sigma_kt_x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
        
        x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
        x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
        
        BS_imp_vol=pd.DataFrame(index = k_array, columns = t_array)
    
        for t in t_array:
        
            c_kt_sigma = self.localVolMC(vanillaoption, locvol, S0, t, k_array, paths, timesteps)
            
            #c_kt_s_sigma_perturb = self.localVolMC(vanillaoption, lambda u, t: locvol(u,t) + self.delta_sigma, S0 + self.delta_S, t, k_array, paths, timesteps)
            c_kt_s_sigma_perturb = self.localVolMC(vanillaoption, lambda u, t: locvol(u,t), S0 + self.delta_S, t, k_array, paths, timesteps, deformation = self.beta)
            
            #c_kt_sigmaperturb = self.localVolMC(vanillaoption, lambda u, t: locvol(u,t) - self.delta_sigma, S0, t, k_array, paths, timesteps)
            
            c_kt_s_perturb = self.localVolMC(vanillaoption, locvol, S0 + self.delta_S, t, k_array, paths, timesteps)
            c_kt_negative_beta = self.localVolMC(vanillaoption, locvol, S0, t, k_array, paths, timesteps, deformation = -self.beta)
            
            volga_call_price_grid[t] = (2 * c_kt_sigma['Value under Local Volatility'] - c_kt_negative_beta['Value under Local Volatility']).values
            vanna_call_price_grid[t] = (c_kt_s_sigma_perturb['Value under Local Volatility'] - c_kt_s_perturb['Value under Local Volatility'] + c_kt_sigma['Value under Local Volatility']).values

            sigma_kt_x_volga_kt_grid[t] = BlackScholes.BSImpliedVol(S0, k_array, t, r, volga_call_price_grid[t].values, OptionType['Call'])
            sigma_kt_x_vanna_kt_grid[t] = BlackScholes.BSImpliedVol(S0, k_array, t, r, vanna_call_price_grid[t].values, OptionType['Call'])
            
            BS_imp_vol[t] = c_kt_sigma['Black Scholes Implied Vol'].values
              
        x_volga_kt_grid= sigma_kt_x_volga_kt_grid - BS_imp_vol
        x_vanna_kt_grid= sigma_kt_x_vanna_kt_grid - BS_imp_vol
                
        self.x_volga_kt = interpolate.RectBivariateSpline(k_array, t_array, x_volga_kt_grid.astype(float).values)
        self.x_vanna_kt = interpolate.RectBivariateSpline(k_array, t_array, x_vanna_kt_grid.astype(float).values)
        
        #For debugging
        self.volga_call_price_grid = volga_call_price_grid
        self.vanna_call_price_grid = vanna_call_price_grid
        
        self.sigma_kt_x_volga_kt_grid = sigma_kt_x_volga_kt_grid
        self.sigma_kt_x_vanna_kt_grid = sigma_kt_x_vanna_kt_grid
        
        self.x_volga_kt_grid = x_volga_kt_grid
        self.x_vanna_kt_grid = x_vanna_kt_grid