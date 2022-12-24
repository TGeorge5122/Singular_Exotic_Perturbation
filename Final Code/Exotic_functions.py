from scipy import interpolate
from scipy.stats import norm
import collections
import numpy as np
import pandas as pd
import pdb
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_df_3d import plot_df_3d
from BSVol_to_LocVol import BSVol_to_LocVol
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

def localVolMC(locvol, S0, T,  paths, timeSteps, AK, nu,kappa,deformation=0,debug = False):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
    Y_var=nu**2/kappa/2
    deformation = np.exp(deformation)
    # We use a vertical array, one element per M.C. path
    s = np.zeros(paths)
    t = 0

    for _ in range(timeSteps):
        
        if debug:
            pdb.set_trace()
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        
        sig = locvol(S0*np.exp(s), t)*deformation
        
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
def ExoticlocalVolMC(locvol, S0, T,  paths, timeSteps, AK,kappa,nu,deformation=0):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)    

    Y_var=nu**2/kappa/2
    deformation = np.exp(deformation)

    # We use a vertical array, one element per M.C. path
    s = np.zeros((timeSteps + 1, paths))
    t = np.zeros(paths)

    for i in range(timeSteps):
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        
        sig = locvol(S0*np.exp(s[i]),t)*deformation
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

def ExoticLSVMC(locvol, S0, T,  paths, timeSteps, AK,kappa,nu,rho):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
    Y_var=nu**2/kappa/2

    # We use a vertical array, one element per M.C. path
    s = np.zeros((timeSteps + 1, paths))
    t = np.zeros(paths)

    Y=nu**2/kappa/2

    for i in range(timeSteps):
        
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1

        W2 = rho*W+np.sqrt(1-rho**2)*np.random.normal(size = paths)
        W2 -= np.mean(W2)
        W2/= np.std(W2,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1

        # Stock SDE discretization
        
        sig = locvol(S0*np.exp(s[i]),t)*np.exp(Y-Y_var)
        s[i+1] = s[i] - sig**2/2*dt + sig*sqdt * W
        Y+=-kappa*Y*dt + nu*sqdt*W2
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