from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq
import scipy.signal
from scipy.signal import welch, periodogram
import os
import sys
sys.path.append('..')
from predict_bp import *
from train.insert_missing import *
import math
import matplotlib.pyplot as plt
import os

n = [4, 6, 8, 10]
pred_horizon = [1, 3, 7]

n_features = 1
method_missing = "MEAN"

data = pd.read_csv('../dataset/myHeartBP.csv', header=None).values
data = insert_missing(data, n, n_features, method_missing)
patients = data  # todos os paciente

# Definindo o valor da linha de hipertensão alta
linha_alta = 140

fs= 1 #0.00001157

densidades_espectrais = []
frequencias = []


# Plotando um gráfico para cada paciente e salvando-o na pasta
frequencias_medias = rfftfreq(60, 1/fs) #len patients e numero de amostras por s
frequencias_medias = frequencias_medias[1:]
for i in range(patients.shape[1]):
    hiper = patients[:,i]

    Pxx = rfft(hiper)
    Pxx = Pxx[1:]
    psd = np.abs(Pxx) / max(Pxx)

    densidades_espectrais.append(psd)

    '''plt.plot(frequencias_medias, np.abs(Pxx))    
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Densidade espectral')
    plt.title('Média do espectro de densidade de glicose para todos os pacientes')
    plt.show()'''

densidades_espectrais_media = np.mean(densidades_espectrais, axis=0)

print("Frequências: ", max(frequencias_medias))
print("PSD: ", max(densidades_espectrais_media))

for valor in densidades_espectrais_media:
    if valor == 0:
        break
    valor_anterior = valor

if valor_anterior is not None:
    print("O valor anterior ao primeiro valor igual a zero é:", valor_anterior)
else:
    print("Não há valores diferentes de zero na lista")

plt.semilogy(frequencias_medias, densidades_espectrais_media)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Densidade espectral')
plt.title('Média do espectro de densidade de hipertensão para todos os pacientes')
plt.show()