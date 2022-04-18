from scipy import interpolate
from scipy.stats import norm
import collections
import numpy as np
import pandas as pd
import pdb

def BSCall (S0, K, T, r, sigma):
    
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

# Bisection method
# This function now works with vectors of strikes and option values
def BSImpliedVolCall (S0, K, T, r, C):
    nK = len(K) if isinstance(K, collections.Sized) else 1
    sigmaL = np.full(nK, 1e-10)
    CL = BSCall(S0, K, T, r, sigmaL)
    sigmaH = np.full(nK, 10)
    CH = BSCall(S0, K, T, r, sigmaH)
        
    while (np.mean(sigmaH - sigmaL) > 1e-10):
        sigma = (sigmaL + sigmaH)/2
        CM = BSCall(S0, K, T, r, sigma)
        CL = CL + (CM < C)*(CM-CL)
        sigmaL = sigmaL + (CM < C)*(sigma-sigmaL)
        CH = CH + (CM >= C)*(CM-CH)
        sigmaH = sigmaH + (CM >= C)*(sigma-sigmaH)
        
    return sigma

def VanillalocalVolMC(locvol, S0, T,  paths, timeSteps, AK):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
  
    # We use a vertical array, one element per M.C. path
    s = np.zeros(paths)
    t = np.zeros(paths)

    for _ in range(timeSteps):
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        
        sig = locvol(S0*np.exp(s), t)
        
        s += -sig**2/2 * dt + sig * sqdt * W
        t += dt
    
    S = S0 * np.exp(s)    
    
    M = len(AK);
    AV, BS_vol = np.zeros(M), np.zeros(M)
    
    # Evaluate mean call value for each path
    for i in range(M):
        
        K = AK[i]
        V = (S > K) * (S - K) # Boundary condition for European call
        AV[i] = np.nanmean(V)
        BS_vol[i] = BSImpliedVolCall(S0, K, T, 0, AV[i])
        
    return pd.DataFrame([AK, AV, BS_vol], index = ['Strike','Local Volatility Price', "Black Scholes Implied Vol"]).T

def ExoticlocalVolMC(locvol, S0, T,  paths, timeSteps, AK):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
  
    # We use a vertical array, one element per M.C. path
    s = np.zeros((timeSteps + 1, paths))
    t = np.zeros(paths)

    for i in range(timeSteps):
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        
        sig = locvol(S0*np.exp(s[i]),t)
        s[i+1] = s[i] - sig**2/2*dt + sig*sqdt * W
        
        t += dt
    
    S = S0 * np.exp(s)    
    
    #Down and out barrier option - barrier at 0.9
    S = (S.min(axis = 0) >= 0.9) * S[-1]
    
    M = len(AK);
    AV = np.zeros(M)
    
    # Evaluate mean call value for each path
    for i in range(M):
        
        K = AK[i]
        V = (S > K) * (S - K) # Boundary condition for European call
        AV[i] = np.nanmean(V)
        
    return pd.DataFrame([AK, AV], index = ['Strike','Local Volatility Price']).T


locvolCEV = lambda u, t: .2/np.sqrt(u)


if __name__ == '__main__':
    
    paths = 5000
    timesteps = 1000
    
    rho = -0.9
    nu = 10
    kappa = 10 
    
    S0 = 1
    r = 0
    
    delta_sigma = 0.03
    delta_S = 0.03
    
    k_array = np.linspace(0.2, 2.4, 10)
    t_array = np.linspace(0, 1, len(k_array))
    t_array[0] = 1/250
    
    volga_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
    sigma_kt_x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    vanna_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
    sigma_kt_x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)

    for t in t_array:
    
        c_kt_sigma = VanillalocalVolMC(locvol = locvolCEV, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        c_kt_s_sigma_perturb = VanillalocalVolMC(locvol = lambda u, t: locvolCEV(u,t) + delta_sigma, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        c_kt_sigmaperturb = VanillalocalVolMC(locvol = lambda u, t: locvolCEV(u,t) - delta_sigma, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        c_kt_s_perturb = VanillalocalVolMC(locvol = locvolCEV, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        
        volga_call_price_grid[t] = (2 * c_kt_sigma['Local Volatility Price'] - c_kt_sigmaperturb['Local Volatility Price']).values
        vanna_call_price_grid[t] = (c_kt_s_sigma_perturb['Local Volatility Price'] - c_kt_s_perturb['Local Volatility Price'] + c_kt_sigma['Local Volatility Price']).values
        
        sigma_kt_x_volga_kt_grid[t] = BSImpliedVolCall(S0, k_array, t, r, volga_call_price_grid[t].values)
        sigma_kt_x_vanna_kt_grid[t] = BSImpliedVolCall(S0, k_array, t, r, vanna_call_price_grid[t].values)
        
    for k in k_array:
        for t in t_array:
            x_volga_kt_grid.at[k,t] = sigma_kt_x_volga_kt_grid.at[k,t] - locvolCEV(k, t)
            x_vanna_kt_grid.at[k,t] = sigma_kt_x_vanna_kt_grid.at[k,t] - locvolCEV(k, t)
            
    x_volga_kt = interpolate.RectBivariateSpline(k_array, t_array, x_volga_kt_grid.astype(float).values)
    x_vanna_kt = interpolate.RectBivariateSpline(k_array, t_array, x_vanna_kt_grid.astype(float).values)

    exotic_volga = pd.DataFrame(index = k_array, columns = t_array)
    exotic_vanna = pd.DataFrame(index = k_array, columns = t_array)
    
    for t in t_array:
        print(t)        
        
        p_kt_sigma_x_vanna = ExoticlocalVolMC(locvol = lambda u, t: locvolCEV(u,t) + x_vanna_kt(u,t,grid=False).flatten(), S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array) 
        p_kt_sigma_x_volga = ExoticlocalVolMC(locvol = lambda u, t: locvolCEV(u,t) + x_volga_kt(u,t,grid=False).flatten(), S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array) 
        p_kt_sigma = ExoticlocalVolMC(locvol = locvolCEV, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        p_kt_s_sigma_perturb = ExoticlocalVolMC(locvol = lambda u, t: locvolCEV(u,t) + delta_sigma, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        p_kt_sigmaperturb = ExoticlocalVolMC(locvol = lambda u, t: locvolCEV(u,t) - delta_sigma, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        p_kt_s_perturb = ExoticlocalVolMC(locvol = locvolCEV, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = k_array)
        
        exotic_volga[t] = p_kt_sigma_x_volga['Local Volatility Price'].values - (2 * p_kt_sigma['Local Volatility Price'] - p_kt_sigmaperturb['Local Volatility Price']).values
        exotic_vanna[t] = (p_kt_s_sigma_perturb['Local Volatility Price'] - p_kt_s_perturb['Local Volatility Price'] + p_kt_sigma['Local Volatility Price']).values - p_kt_sigma_x_vanna['Local Volatility Price'].values
        
    lsv_impact = 1 / 2 * (nu ** 2 / (2 * kappa)) * exotic_volga + rho * nu / kappa * exotic_vanna
