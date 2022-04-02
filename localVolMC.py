# Takes a local volatility surface as input

from BlackScholes import *
import numpy as np
import pandas as pd
import scipy

def localVolMC(locvol, S0, T,  paths, timeSteps, AK):
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
        sig = locvol(S0*np.exp(s),t/2)
        s += -sig**2/2*dt + sig*sqdt * W
        t += dt
        
    M = len(AK);
    AV, AVdev = np.zeros(M), np.zeros(M)
    BSV, BSVH, BSVL = np.zeros(M), np.zeros(M), np.zeros(M)
  
    S = S0*np.exp(s)
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

#------------------------------------------------------------
    # Analytic computation translated from Shaw Chapter 28.
    # Note that nu = 1/(2*beta); beta=1/(2*nu);
    
def CEVfunc(x ,a, nu, n, upper):
    f = lambda z: np.exp(-(z**2/(4*x)))*z**n*scipy.special.iv(nu,z)
    return scipy.integrate.quad(f, a, upper)[0]
    
def CEVCall(S0, K, r, q, vol, t, nu, scale):
    sig = vol*S0**(1/(2*nu))
    if r==q:
        c = 2*nu**2/(sig**2*t)
    else:
        c = 2*nu*(r - q)/(sig**2*(np.exp((r - q)*t/nu) - 1))
    x = c*S0**(1/nu)*np.exp((r - q)*t/nu)
    a = 2*np.sqrt(c*x*K**(1/nu))
    res = np.exp(-r*t - x)*(
            CEVfunc(x, a, nu, nu + 1,scale*a)/(x*c**nu*2**(nu + 1)) - 
            K*((2*x)**(nu-1))*CEVfunc(x, a, nu, 1 - nu, scale*a))
    return res
    
callValue = lambda k: np.array(map(lambda k_: CEVCall(S0=1, K=k_, r=0,q=0, vol=.2, t=1, nu=1, scale=5), k))
impvol = lambda k: BSImpliedVolCall(1,k,1,0,callValue(k))
    
def skewMC(t):
    res = localVolMC(locvolCEV, 1, 1, 500000, 100, np.array([.99,1,1.01]))
    skew = (res['BSV'][3]-res['BSV'][1])/0.02
    return skew
            
def skewAnal(t):
    skew = (impvol(1.01)-impvol(.99))/0.02
    return skew
             
    
locvolCEVdecay = lambda u,t: .2*(1+6*(1/np.sqrt(u)-1)*np.exp(-t))

locvolCEVdecay_faster = lambda u,t: .2*(1+6*(1/np.sqrt(u)-1)*np.exp(-3*t))

    
# Build experiment into MC code
def localVolMCspot(locvol, S0, T,  paths, timeSteps, AK):
    dt = T/timeSteps
    sqdt = np.sqrt(dt)
    S9 = 0.9*S0
      
    # We use a vertical array, one element per M.C. path
    s, s9, t = np.zeros(paths), np.zeros(paths), np.zeros(paths)
    for _ in range(timeSteps):
        W = np.random.normal(size = paths)
        W -= np.mean(W) 
        W /= np.std(W,ddof=1) # As alternative to antithetic variates, force mean<-0 and sd<-1
        # Stock SDE discretization
        sig = locvol(S0*np.exp(s),t/2)
        sig9 = locvol(S9*np.exp(s9),t/2)
        s += -sig**2/2*dt + sig*sqdt * W
        s9 += -sig9**2/2*dt + sig9*sqdt * W
        t += dt
        
    M = len(AK)
    AV, AVdev = np.zeros(M), np.zeros(M)
    BSV, BSVH, BSVL = np.zeros(M), np.zeros(M), np.zeros(M)
    AV2, AVdev2 = np.zeros(M), np.zeros(M)
    BSV2, BSVH2, BSVL2 = np.zeros(M), np.zeros(M), np.zeros(M)
      
    S = S0*np.exp(s)
    S2 = S9*np.exp(s9)
    # Evaluate mean call value for each path
    for i in range(M):
        K = AK[i]
        V = (S>K)*(S - K) # Boundary condition for European call
        V9 = (S2>K)*(S2-K)
        AV[i] = np.nanmean(V)
        AVdev[i] = np.nanstd(V,ddof=1) / np.sqrt(paths)
        BSV[i] = BSImpliedVolCall(S0, K, T, 0, AV[i])
        BSVL[i] = BSImpliedVolCall(S0, K, T, 0, AV[i] - AVdev[i])
        BSVH[i] = BSImpliedVolCall(S0, K, T, 0, AV[i] + AVdev[i])
        AV2[i] = np.nanmean(V9)
        AVdev2[i] = np.nanstd(V9,ddof=1) / np.sqrt(paths)
        BSV2[i] = BSImpliedVolCall(S9, K, T, 0, AV2[i])
        BSVL2[i] = BSImpliedVolCall(S9, K, T, 0, AV2[i] - AVdev2[i])
        BSVH2[i] = BSImpliedVolCall(S9, K, T, 0, AV2[i] + AVdev2[i])
          
    return pd.DataFrame([AK, AV, BSV, AV, BSV2], index = ['AK','AV','BSV', 'AV2', 'BSV2']).T

if __name__ == '__main__':
    
    AK = np.arange(.8,1.3,.1)
    #localVolMC(locvolMJR(subTVSp81), 1, 1, 5000, 1000, AK)
    
    # Test function on CEV
    print(localVolMC(locvolCEV, 1, 1, 5000, 1000, AK))
    
    print(CEVCall(S0=1, K=1, r=0,q=0, vol=.2, t=1, nu=1, scale=5))# This is beta=1/2 which is the square-root model
    
    res11 = localVolMC(locvolCEV, 1, 1, 500000, 1000, np.array([.9,1,1.1]))
    res12 = localVolMC(locvolCEV, .9, 1, 500000, 1000, np.array([.9,1,1.1]))
    betaCEV = (res12['BSV'][0]-res11['BSV'][1])/(res11['BSV'][0]-res11['BSV'][1])
        
    res21 = localVolMC(locvolCEVdecay, 1, 1, 500000, 1000, np.array([.9,1,1.1]))
    res22 = localVolMC(locvolCEVdecay, .9, 1, 500000, 1000, np.array([.9,1,1.1]))
    betaCEVdecay = (res22['BSV'][0]-res21['BSV'][1])/(res21['BSV'][0]-res21['BSV'][1])
        
    print(betaCEV)
    #[1] 1.947534
    print(betaCEVdecay)
    #[1] 2.280621
    # Closed-form approximation gives 2.36
    
    # Repeat experiment with faster decay
    res21_faster = localVolMC(locvolCEVdecay_faster, 1, 1, 500000, 1000, np.array([.9,1,1.1]))
    res22_faster = localVolMC(locvolCEVdecay_faster, .9, 1, 500000, 1000, np.array([.9,1,1.1]))
    betaCEVdecay_faster = (res22_faster['BSV'][0]-res21_faster['BSV'][1])/(res21['BSV'][0]-res21['BSV'][1])
        
    print(betaCEVdecay_faster)
    #[1] 2.718457
    
    res = localVolMCspot(locvolCEVdecay, 1, 1, 500, 1000, np.array([.9,1,1.1]))
    beta = (res['BSV2'][0]-res['BSV'][1])/(res['BSV'][0]-res['BSV'][1])
    
    print(beta)