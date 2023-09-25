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
import warnings
warnings.filterwarnings("ignore", message="Level value of [0-9]+ is too high*")
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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

    scores=[]

    for i in range(0, np.size(patients, 1)):

        #print("------------","PACIENTE -->", i+1, "------------")
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

        # Calcule a derivada da série temporal de glicose
        derivada = np.gradient(filtered_hiper)


############################################################################################################

        X, y = data_prep(filtered_hiper, input_size, pred_horizon, True)
        X_dev, y_dev = data_prep(derivada, input_size, pred_horizon, True)

        X_train = np.concatenate((X, X_dev), axis=1)
        y_train = y


        model = LinearRegression()

        # treinar o modelo com os dados de entrada e saída
        model.fit(X_train, y_train)


        #model.save('models_linear/model_' + str(input_size) + '_' + str(pred_horizon) + '.h5')

        # avaliar o desempenho do modelo em relação aos dados de treinamento
        score = model.score(X_train, y_train)

        scores.append(score)

        y_pred = model.predict(X_train)
        
        fig, ax = plt.subplots()
        plt.plot(y_train, label='Atual')
        plt.plot(y_pred, label='Previsão')
        plt.legend()
        plt.title('Paciente {}'.format(i+1))            
        plt.savefig('graficos_combo/graficos_N10_P7/paciente{}.png'.format(i+1))
        plt.close()

    print("Coeficiente de determinação R²:", np.mean(scores))        
        
    scores.clear()



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

# n = [4, 6, 8, 10]
# pred_horizon = [1, 3, 7]

n = [10]
pred_horizons = [7]

# Criando a pasta para salvar os gráficos
if not os.path.exists('graficos_combo/graficos_N10_P7'):
    os.makedirs('graficos_combo/graficos_N10_P7')

with tf.device('/cpu:0'):
    for input_size in n:
        print("------------","INPUT -->", input_size, "------------")
        for pred_horizon in pred_horizons:
            print("------------","HORIZONTE -->", pred_horizon, "------------")
            train_model(input_size, pred_horizon)
