import tensorflow as tf
import os
from train import insert_missing
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras import Input, Model, backend
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import pywt
import matplotlib.pyplot as plt
import sys

# Defina a frequência de amostragem (amostras por unidade de tempo)
fs = 0.2 # 1 / 5 amostra por minuto

# Defina a frequência de corte do filtro de Butterworth (Hz)
fc = 0.0016 # 1 / 300 Hz

# Defina a ordem do filtro de Butterworth
order = 3




# Defina o tipo de wavelet e o número de escalas desejadas
wavelet = 'db4'
num_scales = 4

threshold = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Por exemplo, para aplicar o filtro na escala 3:
scale_to_filter = 300


n = [4, 6, 8, 10]
pred_horizon = [1, 3, 7]

n_features = 1
method_missing = "MEAN"



# Definindo o valor da linha de hipertensão alta
linha_alta = 140
def derivada(input_size, pred_horizon):

    data = pd.read_csv('dataset/myHeartBP.csv', header=None).values
    data = insert_missing.insert_missing(data, input_size, n_features, method_missing)
    patients = data  # todos os paciente

    # Plotando um gráfico para cada paciente e salvando-o na pasta
    for i in range(0, np.size(patients, 1)):
        print("Paciente", i)
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
        gradiente_significativo = derivada > 0 # vetor booleano indicando valores de gradiente significativos


        # Encontre os índices onde a derivada é positiva e a glicose é acima de 180
        indices_positivos = np.where(derivada > 0)[0]
        momentos_crescentes = [i for i in indices_positivos if filtered_hiper[i] > 140]


        duracao_minima = 24 # duração mínima em minutos
        hiperglicemia_aguda = patients[:, i] > 180 # vetor booleano indicando hiperglicemia aguda
        gradiente_acima_do_limiar = derivada > 0 # vetor booleano indicando valores de gradiente acima do limiar
        gradiente_acima_do_limiar_duracao = np.zeros_like(patients[:, i], dtype=bool) # vetor booleano indicando duração do gradiente acima do limiar
        for i in range(len(gradiente_acima_do_limiar)):
            if gradiente_acima_do_limiar[i] and hiperglicemia_aguda[i]:
                gradiente_acima_do_limiar_duracao[i:i+duracao_minima] = True # considera um episódio de duração mínima

        # Crie o gráfico de linha com marcadores
        plt.plot(filtered_hiper)
        plt.plot(momentos_crescentes, filtered_hiper[momentos_crescentes], 'ro')
        plt.xlabel('Tempo')
        plt.ylabel('Glicose')
        plt.title('Série Temporal de Glicose com Marcadores de Momentos Crescentes Acima de 180')
        plt.show()

        # Crie um vetor de tempo para a série temporal de glicose
        tempo = np.arange(len(filtered_hiper))

        # Crie um gráfico com a série temporal de glicose e seus pontos derivados
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Tempo (minutos)')
        ax1.set_ylabel('Glicose', color=color)
        ax1.plot(tempo, filtered_hiper, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Derivada', color=color)
        ax2.plot(tempo, derivada, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        

        fig.tight_layout()
        plt.show()



    
for input_size in [*range(3, 11)]:
    for pred_horizon in [1, 3, 7]:
        derivada(input_size, pred_horizon)