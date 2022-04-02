from BlackScholes import *
import numpy as np
import pandas as pd

def barrier_localVolMC(locvol, S0, T, B, AK, paths, timeSteps):
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
  
    # We use a vertical array, one element per M.C. path
    s = np.zeros(paths)
    t = np.zeros(paths)
    cross_barrier = np.ones(paths)
    
    for _ in range(timeSteps):
        W = np.random.normal(size = paths)
        W -= np.mean(W)
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        S = S0*np.exp(s)
        cross_barrier = (S <= B) * cross_barrier
        sig = locvol(S,t/2)
        s += -sig**2/2*dt + sig*sqdt * W
        t += dt
        
    M = len(AK);
    AV, AVdev = np.zeros(M), np.zeros(M)
    BSV, BSVH, BSVL = np.zeros(M), np.zeros(M), np.zeros(M)
  
    S = S0*np.exp(s) * cross_barrier
    # Evaluate mean call value for each path
    for i in range(M):
        K = AK[i]
        V = (S > K) * (S - K) # Boundary condition for European call
        AV[i] = np.nanmean(V)
        AVdev[i] = np.nanstd(V,ddof=1) / np.sqrt(paths)
        BSV[i] = BSImpliedVolCall(S0, K, T, 0, AV[i])
        BSVL[i] = BSImpliedVolCall(S0, K, T, 0, AV[i] - AVdev[i])
        BSVH[i] = BSImpliedVolCall(S0, K, T, 0, AV[i] + AVdev[i])
        
    return pd.DataFrame([AK, AV, BSV], index = ['AK','AV','BSV']).T
 
locvolCEV = lambda u, t: .2/np.sqrt(u)

if __name__ == '__main__':
    
    AK = np.arange(.8,1.3,.1)
    
    # Test function on CEV
    print(barrier_localVolMC(locvolCEV, 1, 1, 1.4, AK, 5000, 1000))