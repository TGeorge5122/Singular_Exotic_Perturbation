import numpy as np
def lvSPX(mintau,S0):
    
    a = 0.0012
    b =  0.1634
    sig =  0.1029
    rho = -0.5555
    m = 0.0439
    
    # Local volatility
    
    def lv(S,t):
        k=np.log(S/S0)
        t = np.maximum(t,mintau)
        kp = k/np.sqrt(t)
        v = a + b * (rho * (kp-m) + np.sqrt((kp-m)**2 + sig**2 * t))
        return np.sqrt(abs(v))
        
    return lv