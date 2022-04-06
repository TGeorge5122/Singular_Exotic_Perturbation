import numpy as np
import collections
from enum import Enum
from scipy.stats import norm

OptionType = Enum('OptionType', ['Call','Put'])

class BlackScholes:

    @staticmethod
    def BSFormula (S0, K, T, r, sigma, optiontype: OptionType):
        
        x = np.log(S0 / K) + r * T
        sig = sigma * np.sqrt(T)
        d1 = x / sig + sig / 2
        d2 = d1 - sig
        pv = np.exp(-r * T)
        return S0 * norm.cdf(d1) - pv * K * norm.cdf(d2) if optiontype == \
            OptionType['Call'] else pv * K * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    @staticmethod
    def BSImpliedVol(S0, K, T, r, V, optiontype: OptionType):
        
        nK = len(K) if isinstance(K, collections.Sized) else 1
        sigmaL = np.full(nK, 1e-10)
        VL = BlackScholes.BSFormula(S0, K, T, r, sigmaL, optiontype)
        sigmaH = np.full(nK, 10.0)
        VH = BlackScholes.BSFormula(S0, K, T, r, sigmaH, optiontype)
        
        while (np.mean(sigmaH - sigmaL) > 1e-10):
            sigma = (sigmaL + sigmaH)/2
            VM = BlackScholes.BSFormula(S0, K, T, r, sigma, optiontype)
            VL += (VM < V) * (VM - VL)
            sigmaL += (VM < V) * (sigma - sigmaL)
            VH += (VM >= V) * (VM - VH)
            sigmaH += (VM >= V) * (sigma - sigmaH)
            
        return sigma
    
    @staticmethod
    def BSImpliedVol_OTM(S0, K, T, r, V):
      return np.where(K > S0 * np.exp(r*T), BlackScholes.BSImpliedVol(S0, K, T,\
            r, V, OptionType['Call']),  BlackScholes.BSImpliedVol(S0, K, T, r, \
            V, OptionType['Put']))
