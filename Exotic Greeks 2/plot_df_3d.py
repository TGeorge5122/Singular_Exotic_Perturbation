# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_df_3d(df,xlabel='time',ylabel='strike',zlabel='volatility',title='surface_plot'):
    '''
    Given a df, plot 3d surface of this df
    '''
    
    
    # Creating dataset
    y = list(df.index)
    x = list(df.columns)
    X, Y = np.meshgrid(x, y)
    Z=df.values
    
    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    
    # Creating color map
    my_cmap = plt.get_cmap('winter')
    
    # Creating plot
    surf = ax.plot_surface(X, Y, Z,
                        cmap = my_cmap,
                        edgecolor ='none')
    
    fig.colorbar(surf, ax = ax,
                shrink = 0.5, aspect = 5)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # show plot
    plt.show()
