import numpy as np

def insert_missing(data):

    # substituicao dos valores em falta pela media dos valores nao nulos de cada paciente
    for i in range(data.shape[1]):
        mean = np.nanmean(data.iloc[:,i])
        idx = np.where(np.isnan(data.iloc[:,i]))
        data.iloc[idx, i] = mean    

    return data
