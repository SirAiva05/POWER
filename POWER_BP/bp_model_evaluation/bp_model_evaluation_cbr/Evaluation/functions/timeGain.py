import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def timeGain(y, y_hat, p):
    
    if(np.size(y, 1)==1):
        y = pd.transpose(y)
        y_hat = pd.transpose(y_hat)
        
    Y = np.matlib.repmat(y, 1, p+1)
    Y = Y[1:-1-p,:] 
    Y_HAT = np.zeros(len(y)-p, p+1)
    for i in range(0,p):
        Y_HAT[:, i+1] = y_hat[i+1:-1-(p-i)]
        if (0):
            plt.plot(range(1,-1-p), 'r--')
            plt.show()
            plt.plot(range(i+1,-1-(p-i)), 'bo')
            plt.pause()
            plt.clf()
            
    delay = Y - Y_HAT
    delay = delay ** 2
    _,delay = min(delay)
    delay = delay-1
    TG = p - delay
    
    return TG
