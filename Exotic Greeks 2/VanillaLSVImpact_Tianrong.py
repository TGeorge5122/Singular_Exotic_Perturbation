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

from SVI_localvol import lvSPX
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

def localVolMC(locvol, S0, T,  paths, timeSteps, AK, deformation=0, debug = False):
    
    dt = T / timeSteps
    sqdt = np.sqrt(dt)
    deformation=np.exp(deformation)
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

locvolCEV = lambda u, t: .2/np.sqrt(u)





if __name__ == '__main__':
    
    paths = 500
    timesteps = 1000
    
    rho = 0
    nu = 0.1#10
    kappa = 10#10 
    
    S0 = 1
    r = 0
    locvolMLP=lvSPX(mintau=1/250,S0=S0)
    delta_sigma = 0.03
    delta_S = 0.03
    beta = 0.03#deformation term
    k_array = np.linspace(-0.6, 0.2, 10)#log stock price log(St/S0)
    Strike_array=S0*np.exp(k_array)# actual stock price St
    t_array = np.linspace(0, 1, len(k_array))
    t_array[0] = 1/250
    

    volga_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
    sigma_kt_x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    x_volga_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    vanna_call_price_grid = pd.DataFrame(index = k_array, columns = t_array)
    sigma_kt_x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    x_vanna_kt_grid = pd.DataFrame(index = k_array, columns = t_array)
    BS_imp_vol=pd.DataFrame(index = k_array, columns = t_array)

    Call_LV=pd.DataFrame(index = k_array, columns = t_array)

    
    LVsurface=pd.DataFrame(index = Strike_array, columns = t_array,dtype=float)
    for t in t_array:
        for K in Strike_array:
            LVsurface.loc[K,t]=float(locvolMLP(K,t))
    plot_df_3d(LVsurface,title='LV surface')

    for t in t_array:

        c_kt_sigma = localVolMC(locvol = locvolMLP, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array)
        c_kt_s_sigma_perturb = localVolMC(locvol = lambda u, t: locvolMLP(u,t) + delta_sigma, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array)
        c_kt_sigmaperturb = localVolMC(locvol = lambda u, t: locvolMLP(u,t) - delta_sigma, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array)
        c_kt_s_perturb = localVolMC(locvol = locvolMLP, S0 = S0 + delta_S, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array)
        
        #deformation with e^beta
        c_kt_negative_beta=localVolMC(locvol = locvolMLP, S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array,deformation=-beta)
        Call_LV[t]=c_kt_sigma['Local Volatility Price'].values
        volga_call_price_grid[t] = (2 * c_kt_sigma['Local Volatility Price'] - c_kt_negative_beta['Local Volatility Price']).values
        vanna_call_price_grid[t] = (c_kt_s_sigma_perturb['Local Volatility Price'] - c_kt_s_perturb['Local Volatility Price'] + c_kt_sigma['Local Volatility Price']).values
        
        sigma_kt_x_volga_kt_grid[t] = BSImpliedVolCall(S0, Strike_array, t, r, volga_call_price_grid[t].values)
        sigma_kt_x_vanna_kt_grid[t] = BSImpliedVolCall(S0, Strike_array, t, r, vanna_call_price_grid[t].values)
        BS_imp_vol[t]=c_kt_sigma['Black Scholes Implied Vol'].values
    # for k in k_array:
    #     for t in t_array:
    #         # change locvol to BS impvol
    #         x_volga_kt_grid.at[k,t] = sigma_kt_x_volga_kt_grid.at[k,t] - BS_imp_vol.at[k, t]
    #         x_vanna_kt_grid.at[k,t] = sigma_kt_x_vanna_kt_grid.at[k,t] - BS_imp_vol.at[k, t]
    volga_LocalVol=BSVol_to_LocVol(sigma_kt_x_volga_kt_grid)
    vanna_LocalVol=BSVol_to_LocVol(sigma_kt_x_vanna_kt_grid)
    
    volga_LocalVol_df=pd.DataFrame(index = k_array, columns = t_array).apply(lambda x:volga_LocalVol(x.index , x.name))
    vanna_LocalVol_df=pd.DataFrame(index = k_array, columns = t_array).apply(lambda x:vanna_LocalVol(x.index , x.name))

    x_volga_kt_grid= volga_LocalVol_df- LVsurface
    x_vanna_kt_grid= vanna_LocalVol_df- LVsurface

       
    #t_interp = []
    #for _ in range(len(k_array)):
    #    t_interp.extend(t_array)
     
    x_volga_kt = interpolate.RectBivariateSpline(Strike_array, t_array, x_volga_kt_grid.astype(float).values)
    x_vanna_kt = interpolate.RectBivariateSpline(Strike_array, t_array, x_vanna_kt_grid.astype(float).values)
    plot_df_3d(sigma_kt_x_volga_kt_grid,xlabel='time',ylabel='log strike',zlabel='volatility',title='sig_volga_kt')
    plot_df_3d(BS_imp_vol,xlabel='time',ylabel='log strike',zlabel='volatility',title='sig_BS_kt')

    plot_df_3d(x_volga_kt_grid.iloc[2:,2:],xlabel='time',ylabel='log strike',zlabel='volatility',title='x_volga_kt')
    plot_df_3d(x_vanna_kt_grid)
    
    #x_volga_kt = interpolate.interp2d(np.repeat(k_array, len(t_array)), t_interp, [float(x) for x in x_volga_kt_grid.values.flatten()], kind='cubic')
    #x_vanna_kt = interpolate.interp2d(np.repeat(k_array, len(t_array)), t_interp, [float(x) for x in x_vanna_kt_grid.values.flatten()], kind='cubic')
    
    exotic_volga = pd.DataFrame(index = k_array, columns = t_array)
    exotic_vanna = pd.DataFrame(index = k_array, columns = t_array)
    
    for t in t_array:
        print(t)        
        
        c_kt_sigma_x_vanna = localVolMC(locvol = lambda u, t: locvolMLP(u,t) + x_vanna_kt(u,t,grid=False).flatten(), S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array, debug = False) 
        c_kt_sigma_x_volga = localVolMC(locvol = lambda u, t: locvolMLP(u,t) + x_volga_kt(u,t,grid=False).flatten(), S0 = S0, T = t,  paths = paths, timeSteps = timesteps, AK = Strike_array) 
        exotic_volga[t] = (c_kt_sigma_x_volga['Local Volatility Price'].values - volga_call_price_grid[t].values)/beta**2
        exotic_vanna[t] = (vanna_call_price_grid[t].values - c_kt_sigma_x_vanna['Local Volatility Price'].values)/(delta_S*delta_sigma)*S0
    print(np.abs(exotic_volga).mean().mean())
    print(np.abs(exotic_vanna).mean().mean())
    plot_df_3d(vanna_call_price_grid)
    plot_df_3d(exotic_volga,title="Exotic volga")
    lsv_impact = 1 / 2 * (nu ** 2 / (2 * kappa)) * exotic_volga + rho * nu / kappa * exotic_vanna
    plot_df_3d(lsv_impact,title="LSV impact")
    plot_df_3d(lsv_impact.div(Call_LV).iloc[:-3,:-3],title="LSV impact over Call price")
    1==1
