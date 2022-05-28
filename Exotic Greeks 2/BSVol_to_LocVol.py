import pandas as pd
from collections.abc import Callable
from scipy import interpolate
from plot_df_3d import plot_df_3d

def BSVol_to_LocVol( BSVol : pd.DataFrame) -> Callable[[float,float],float]:
    '''
    Suppose the input columns are time and index is for log strikes
    '''
    t_array=BSVol.columns
    y_array=BSVol.index
    w=BSVol.copy()
    for t in t_array:
        w[t]=w[t]**2*t
    dwdt=w.copy()
    dwdy=w.copy()
    dwdy2=w.copy()
    # Getting dW/dT [:,0:L]
    for i in range(len(t_array)-1):
        dwdt.iloc[:,i]=(dwdt.iloc[:,i+1]-dwdt.iloc[:,i])/(t_array[i+1]-t_array[i])
    # Getting dW/dy [0:L,]
    for i in range(len(y_array)-1):
        dwdy.iloc[i,:]=(dwdy.iloc[i+1,:]-dwdy.iloc[i,:])/(y_array[i+1]-y_array[i])
    # Getting second derivative d^2 W/dy^2 [0:L-1,:]
    for i in range(len(y_array)-2):
        dwdy2.iloc[i,:]=(dwdy2.iloc[i+1,:]-dwdy2.iloc[i,:])/((y_array[i+2]-y_array[i])/2)
    # As a result, the valid grid range is [0:L-1,0:L]
    locVol=w.copy()
    for i in range(len(y_array)-2):
        for j in range(len(t_array)-1):
            locVol.iloc[i,j]=max(0,(dwdt.iloc[i,j]/(1-(y_array[i]/w.iloc[i,j])*dwdy.iloc[i,j]+0.5*dwdy2.iloc[i,j]\
                +0.25*(y_array[i]**2/w.iloc[i,j]**2-1/w.iloc[i,j]-0.25)*dwdy.iloc[i,j]**2)))
    # Dupire formula for total variance
    plot_df_3d(BSVol,title='BS vol')
    
    # locVol=dwdt/(1-(y_array/TotalVar)*dwdy+0.5*dwdy2+0.25*(y_array**2/TotalVar**2-1/TotalVar-0.25)*dwdy**2)
    plot_df_3d(locVol,title='loval vol')
    #interpolate and extrapolate
    locVol_fun = interpolate.RectBivariateSpline(y_array, t_array, locVol.astype(float).values)
    return locVol_fun