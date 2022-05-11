import numpy as np
import pandas as pd
import pdb
from SVI_localvol import lvSPX

def localStoVolMC(locvol, S0, T,  paths, timeSteps, AK, kappa, nu, rho, epsilon = 1, debug = False):
    
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
    s, Y_t = np.zeros(paths), np.zeros(paths)
    t = 0

    for _ in range(timeSteps):
        
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
        
        sig = locvol(S0*np.exp(s), t) * np.exp(Y_t) / f_adjustment(t)
        
        Y_t = np.exp(-theta * dt) * Y_t + sigma * sqdt * B        
        
        s += -sig**2/2 * dt + sig * sqdt * W
        t += dt
    
    S = S0 * np.exp(s)    
    
    M = len(AK);
    AV = np.zeros(M)
    
    # Evaluate mean call value for each path
    for i in range(M):
        
        K = AK[i]
        V = (S > K) * (S - K) # Boundary condition for European call
        AV[i] = np.nanmean(V)
        
    return pd.DataFrame([AK, AV], index = ['Strike','Local Volatility Price']).T

def localVolMC(locvol, S0, T,  paths, timeSteps, AK, debug = False):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
    # We use a vertical array, one element per M.C. path
    s = np.zeros(paths)
    t = 0

    for _ in range(timeSteps):
        
        if debug:
            pdb.set_trace()
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1)
        
        sig = locvol(S0*np.exp(s), t)
        
        s += -sig**2/2 * dt + sig * sqdt * W
        t += dt
    
    S = S0 * np.exp(s)    
    
    M = len(AK);
    AV = np.zeros(M)
    
    # Evaluate mean call value for each path
    for i in range(M):
        
        K = AK[i]
        V = (S > K) * (S - K) # Boundary condition for European call
        AV[i] = np.nanmean(V)
        
    return pd.DataFrame([AK, AV], index = ['Strike','Local Volatility Price']).T


if __name__ == '__main__':
    
    paths = 5000
    timesteps = 1000
    
    rho = -0.9
    nu = 1
    kappa = 10 
    
    S0 = 1
    T = 1
    r = 0
    locvolMLP=lvSPX(mintau=1/250,S0=S0)
    
    k_array = np.linspace(-0.6, 0.2, 10)#log stock price log(St/S0)
    Strike_array=S0*np.exp(k_array)
    t_array = np.linspace(0, 1, len(k_array))
    t_array[0] = 1/250
    
    stochastic_lv_results = localStoVolMC(locvolMLP, S0, T,  paths, timesteps, Strike_array, kappa, nu, rho)
    original_lv_results = localVolMC(locvolMLP, S0, T,  paths, timesteps, Strike_array)
    
    print(stochastic_lv_results)
    print(original_lv_results)