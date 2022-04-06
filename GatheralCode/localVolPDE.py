# Adapted from Rolf Poulsen's code
# In this version, sigma is a function of S and t

from BlackScholes import *
import scipy
import pandas as pd
import numpy as np

def fdLocalVol(S0,capT,strike,r,sigma,dS,dt,sdwidth=4, theta=0.5, start=1):
    
    delta = 0 # We can reinstate dividends later...
    mu = r-delta
    
    def tridagsoln(a,b,c,r):
        n = len(b)
        gam, u = np.arange(n,dtype = 'float64'), np.arange(n,dtype = 'float64')
        bet = b[0]
        u[0] = r[0]/bet
        for j in range(1,n):
            gam[j] = c[j-1]/bet
            bet = b[j]-a[j]*gam[j]
            u[j] = (r[j]-a[j]*u[j-1])/bet
        
        for j in range(n-2,-1,-1): u[j] -= gam[j+1]*u[j+1]
    
        return u
    
    #sVec <- S0+(strike-S0)*(1:10)/10;
    #tVec <- (1:10)/10*capT;
    sigmaAvg = (sigma(S0,0)+sigma(strike,capT))/2 # Compute rough MLP estimate
    
    initialcond = lambda S: np.maximum(S-strike,0)
    
    highboundary = lambda S,timeleft: S*np.exp(-delta*timeleft)-np.exp(-r*timeleft)*strike
    lowboundary = lambda S,timeleft: 0
    
    # set up finite-difference grid
    
    Smax = S0*(1+sdwidth*sigmaAvg*np.sqrt(capT))
    Smin = max(2*dS,S0*(1-sdwidth*sigmaAvg*np.sqrt(capT)))
    
    Sgrid = [strike]
    
    while (max(Sgrid) < Smax): Sgrid.append(max(Sgrid)+dS)
    while (min(Sgrid) > Smin): Sgrid.insert(0,min(Sgrid)-dS)
    
    tvec = [capT]
    while (min(tvec) > 0): tvec.insert(0,max(tvec[0]-dt,0))
    tvec = np.array(tvec)
    
    Smin = min(Sgrid) - dS
    Smax = max(Sgrid) + dS
    
    n_space = len(Sgrid)
    n_time = len(tvec)
    
    result = np.full((n_space+2,n_time),np.nan)# Note that the whole grid is stored in memory!
    
    SGridPlus = np.array([Smin] + Sgrid + [Smax])
    Sgrid = np.array(Sgrid)
    result[:,n_time-1] = initialcond(SGridPlus)
    tm1 = capT-dt+dt*theta # Time computed consistently with implicitness
    if (start==1): result[:,(n_time-2)] = BSFormula(SGridPlus, strike, dt, r, sigma(SGridPlus,tm1))
    
    result[0,:] = lowboundary(Smin,capT-tvec)
    result[(n_space+1),:] = highboundary(Smax,capT-tvec)
    
    for j in range(n_time-2-start,-1,-1):
    
        dt = tvec[j+1]-tvec[j]
        t1 = tvec[j]+theta*dt # Time chosen to be consistent with implicitness parameter theta
        
        # Note that these vectors are now time-dependent in general and need to be inside the time-loop
        vol = sigma(Sgrid,t1)
        a = ((1-theta)/(2*dS))*(mu*Sgrid-(vol*Sgrid)**2/dS)
        b = 1/dt+(1-theta)*(r+(vol*Sgrid/dS)**2)
        c = ((1-theta)/(2*dS))*(-mu*Sgrid-(vol*Sgrid)**2/dS)
    
        alpha = ((-theta)/(2*dS))*(mu*Sgrid-(vol*Sgrid)**2/dS)
        beta = 1/dt-theta*(r+(vol*Sgrid/dS)**2)
        gamma = (theta/(2*dS))*(mu*Sgrid+(vol*Sgrid)**2/dS)
        eps = 0
    
        RHS = alpha*result[:n_space,j+1]+beta*result[1:n_space+1,j+1]+gamma*result[2:(n_space+2),j+1]
    
        RHS[0] -= a[0]*result[0,j]
        RHS[n_space-1] = RHS[n_space-1]-c[n_space-1]*result[(n_space+1),j]
    
        result[1:(n_space+1),j] = tridagsoln(a,b,c,RHS)
    
    return pd.DataFrame([SGridPlus, tvec], index = ['spacegrid','timegrid']).T, result

#------------------------------------------------------------------------------------------
# Generic version of PDE code
def callLocalVolPDE(S0, K, r, q, sigma, t, dS, dt, sdw, start=1, theta=1/2):
    
    tst, soln = fdLocalVol(S0, t, K,r,sigma,2*dS,dt,sdw,start=start,theta=theta)
    Sout = tst['spacegrid'].dropna().values
    Sout = Sout[abs(Sout-S0)<4*dS]
    w2h1k = scipy.interpolate.interp1d(tst['spacegrid'].dropna().values, soln[:,0])(Sout)

    tst, soln = fdLocalVol(S0, t, K,r,sigma,dS,dt,sdw,start=start, theta=theta)
    w1h1k = scipy.interpolate.interp1d(tst['spacegrid'].dropna().values, soln[:,0])(Sout)
    
    tst, soln = fdLocalVol(S0, t, K,r,sigma,dS,2*dt,sdw,start=start,theta=theta)
    w1h2k = scipy.interpolate.interp1d(tst['spacegrid'].dropna().values, soln[:,0])(Sout)

    wextrapol = w1h1k+(1/3)*(w1h1k-w1h2k)+(1/3)*(w1h1k-w2h1k) # Richardson extrapolation
    return scipy.interpolate.interp1d(Sout, wextrapol)(S0)
