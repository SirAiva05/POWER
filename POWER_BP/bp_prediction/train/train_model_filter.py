import tensorflow as tf
import os
from insert_missing import insert_missing
from data_prep import data_prep
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras import Input, Model, backend
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import pywt

import sys



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Por exemplo, para aplicar o filtro na escala 3:
scale_to_filter = 300


# treino de modelos JNN para predicao consoante o tamanho do input e o horizonte de predicao
def train_model(input_size, pred_horizon):

    # PARAMS
    activationfcn = 'sigmoid' 
    activationfcn1 = 'linear'
    optimizerfcn = 'adam'
    lossfcn = 'mse'
    n_features = 1
    units = 4
    units1 = 1

    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data = pd.read_csv(parent + '/dataset/myHeartBP.csv', header=None).values
    data = insert_missing(data, input_size, n_features, 'MEAN')


    patients = data  # todos os pacientes

    # input layer
    inputs = Input(shape=(input_size, n_features))
    inputs = Flatten()(inputs)

    # passar o input para a hidden layer - parte nao linear
    dense = Dense(units=4, activation='sigmoid')
    hidden = Flatten()(dense(inputs))
    
    # obter o 1 output (o da camada de input) e juntar ao 2 output (o primeiro output esta a saltar a hidden layer) - parte linear
    jump = Concatenate(axis=1)([inputs, hidden])
    outputs = Dense(units=1, activation='linear')(jump) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizerfcn, loss=lossfcn)

    for i in range(0, np.size(patients, 1)):


############################################################################################################
        
        # Calcule os coeficientes do filtro de Butterworth
        b, a = butter(order, 2*fc/fs, btype='lowpass', analog=False)

        # Aplique o filtro de Butterworth na série temporal de níveis de glicose
        hiper_filtrada = filtfilt(b, a, patients[:, i])



        # Realize a decomposição da série em diferentes escalas usando a transformada wavelet
        coefficients = pywt.wavedec(hiper_filtrada, wavelet, level=num_scales)

        # Aplique o filtro em uma ou mais escalas selecionadas
        filtered_coeffs = [soft_thresholding(c, threshold) for c in coefficients[1:]]
        #coefficients[scale_to_filter] = pywt.threshold(coefficients[scale_to_filter], 2, mode='soft')

        # Reconstrua a série filtrada usando a transformada wavelet inversa
        #filtered_glicose = pywt.waverec(coefficients, wavelet)
        filtered_hiper = pywt.waverec([coefficients[0]] + filtered_coeffs, wavelet)

############################################################################################################

        X, y = data_prep(filtered_hiper, input_size, pred_horizon, True)


        if X.size != 0 and y.size != 0:
            # reshape de [padroes, n] para [padroes, n, n_features]
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            model.fit(X, y, epochs=300, verbose=0)


    model_path = parent + '/models/' + str(input_size) + \
        '_' + str(pred_horizon) 
    model.save(model_path)
    print("Model --> ", model_path)

    backend.clear_session()







# Defina a ordem do filtro de Butterworth
order = 3

# Defina a frequência de amostragem (amostras por unidade de tempo)
fs = 1 # 1 amostra por dia

# Defina a frequência de corte do filtro de Butterworth (Hz)
fc = 0.33 # 1 / 3 Hz

# Defina o tipo de wavelet e o número de escalas desejadas
wavelet = 'db4'
num_scales = 4

threshold = 10


with tf.device('/cpu:0'):
    for input_size in [*range(3, 11)]:
        for pred_horizon in [1, 3, 7]:
            train_model(input_size, pred_horizon)
