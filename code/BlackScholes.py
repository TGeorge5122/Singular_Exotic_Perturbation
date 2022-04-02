import numpy as np
import pandas as pd
import collections
from scipy.stats import norm

def BSFormula (S0, K, T, r, sigma):
    x = np.log(S0/K)+r*T
    sig = sigma*np.sqrt(T)
    d1 = x/sig+sig/2
    d2 = d1 - sig
    pv = np.exp(-r*T)
    return S0*norm.cdf(d1) - pv*K*norm.cdf(d2)

def BSFormulaPut (S0, K, T, r, sigma):
    x = np.log(S0/K) + r * T
    sig = sigma * np.sqrt(T)
    d1 = x/sig + sig/2
    d2 = d1 - sig
    pv = np.exp(-r * T)
    return pv * K * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# This function now works with vectors of strikes and option values
def BSImpliedVolCall (S0, K, T, r, C):
    nK = len(K) if isinstance(K, collections.Sized) else 1
    sigmaL = np.full(nK, 1e-10)
    CL = BSFormula(S0, K, T, r, sigmaL)
    sigmaH = np.full(nK, 10)
    CH = BSFormula(S0, K, T, r, sigmaH)
        
    while (np.mean(sigmaH - sigmaL) > 1e-10):
        sigma = (sigmaL + sigmaH)/2
        CM = BSFormula(S0, K, T, r, sigma)
        CL = CL + (CM < C)*(CM-CL)
        sigmaL = sigmaL + (CM < C)*(sigma-sigmaL)
        CH = CH + (CM >= C)*(CM-CH)
        sigmaH = sigmaH + (CM >= C)*(sigma-sigmaH)
        
    return sigma
    

# This function also works with vectors of strikes and option values  
def BSImpliedVolPut(S0, K, T, r, P):
    sigmaL = 1e-10
    PL = BSFormulaPut(S0, K, T, r, sigmaL)
    sigmaH = 10
    PH = BSFormulaPut(S0, K, T, r, sigmaH)
    while(np.mean(sigmaH - sigmaL) > 1e-10):
        sigma = (sigmaL + sigmaH)/2
        PM = BSFormulaPut(S0, K, T, r, sigma)
        PL = PL + (PM < P) * (PM - PL)
        sigmaL = sigmaL + (PM < P) * (sigma - sigmaL)
        PH = PH + (PM >= P) * (PM - PH)
        sigmaH = sigmaH + (PM >= P) * (sigma - sigmaH)
    return sigma

def BSImpliedVol_OTM(S0, K, T, r, V):
  f = S0 * np.exp(r*T)
  return np.where(K > f, BSImpliedVolCall(S0, K, T, r, V), BSImpliedVolPut(S0, K, T, r, V))


# Function to compute option prices and implied vols given list of final values of underlying
def bsOut(xf, T, AK):
    nK = len(AK)
    N = len(xf)
    xfbar = np.mean(xf)
    CAV = np.zeros(nK)
    BSV = np.zeros(nK)
    BSVL = np.zeros(nK)
    BSVH = np.zeros(nK)
    err = np.zeros(nK)
    for j in range(nK):
        payoff = (xf - AK[j]) * (xf > AK[j])
        CAV[j] = sum(payoff)/N
        err[j] = np.sqrt(np.var(payoff,ddof=1)/N) #ddof=1 uses n-1, np.var default uses n
        #err[j] = np.sqrt(np.var(payoff)/N)
        BSV[j] = BSImpliedVolCall(xfbar, AK[j], T, 0, CAV[j])
        BSVL[j] = BSImpliedVolCall(xfbar, AK[j], T, 0, CAV[j] - err[j])
        BSVH[j] = BSImpliedVolCall(xfbar, AK[j], T, 0, CAV[j] + err[j])
    return pd.DataFrame([AK, CAV, BSV, BSVL, BSVH, err], index = ['AK','CAV','BSV','BSVL', 'BSVH', 'err']).T

# Function to return implied vols for a range of strikes
def analyticOut(callFormula,AK,T):
    nK = len(AK)
    #callFormula is a function that computes the call price
    callPrice = np.zeros(nK)
    BSV = np.zeros(nK)
    for j in range(nK):
        callPrice[j] = callFormula(AK[j])
        BSV[j] = BSImpliedVolCall(1, AK[j], T,0, callPrice[j])
    return pd.DataFrame([AK, callPrice, BSV], index = ['AK','callPrice','BSV']).T
