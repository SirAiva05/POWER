import numpy as np

def ESODnorm (y, y_hat):
    
    x = y
    L = len(y)
    ESOD = []
    for i in range(3, L + 1):
        ESOD.append((x[i] - 2 * x[i-1] + x[i-2]) ** 2)
    ESOD_y = np.sum(ESOD)
    
    x = y_hat
    L = len(y)
    ESOD = []
    for i in range(3, L + 1):
        ESOD.append((x[i] - 2 * x[i-1] + x[i-2]) ** 2)
    ESOD_y_hat = np.sum(ESOD)
    
    ESODnormalized = ESOD_y_hat/ESOD_y
    
    return ESODnormalized
