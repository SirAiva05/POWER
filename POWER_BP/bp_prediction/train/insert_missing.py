import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten

#from data_prep import data_prep


def insert_missing(data, n, n_features, method_missing):

    for patient_idx in range(data.shape[1]):

        patient_data = data[:, patient_idx]
        
        # substituicao dos missing values de cada paciente pela media dos valores nao nulos do mesmo        
        if method_missing == 'MEAN':
            mean = np.nanmean(patient_data)
            idx = np.where(np.isnan(patient_data))
            patient_data[idx] = mean
        
        data[:, patient_idx] = patient_data

    return data
