from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
def SVI(xdata,a,b,sig,rho,m):
    '''
    xdata is m*2 dimentional
    '''
    mintau=1/250
    
    k=xdata.get_level_values(0)
    t=xdata.get_level_values(1)
    t = np.maximum(t,mintau)
    kp = k/np.sqrt(t)
    v = a + b * (rho * (kp-m) + np.sqrt((kp-m)**2 + sig**2 * t))
    return np.sqrt(abs(v))
def SVI_params(a,b,sig,rho,m,mintau=1/250):
    def lv(k,t):
        t = np.maximum(t,mintau)
        kp = k/np.sqrt(t)
        v = a + b * (rho * (kp-m) + np.sqrt((kp-m)**2 + sig**2 * t))
        return np.sqrt(abs(v))
    return lv
def SVIFit(df):
    '''
    Input is a dataframe of localVol
    Return SVI fits
    '''

    SVIparams=np.array([0.0012,0.1634,0.1029,-0.5555,0.0439])
    data=df.stack()
    data=pd.concat([data,data.loc[np.abs(data.index.get_level_values(0))<0.02]])## put higher weight on ATM options
    data=pd.concat([data,data.loc[np.abs(data.index.get_level_values(0))<0.02]])## put higher weight on ATM options
    popt, pcov=curve_fit(SVI,data.index,data.values,p0=SVIparams)
    return popt




