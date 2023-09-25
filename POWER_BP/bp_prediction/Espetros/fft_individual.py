from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import scipy.signal
from scipy.signal import welch, periodogram
import os

from predict_bp import *
from train.insert_missing import * 
import math
import matplotlib.pyplot as plt
import os

n = [4, 6, 8, 10]
pred_horizon = [1, 3, 7]

n_features = 1
method_missing = "MEAN"

data = pd.read_csv('dataset/myHeartBP.csv', header=None).values
data = insert_missing(data, n, n_features, method_missing)
patients = data  # todos os paciente

# Definindo o valor da linha de hipertensão alta
linha_alta = 140

fs= 86.400

# Criando a pasta para salvar os gráficos
if not os.path.exists('graficos_densidade'):
    os.makedirs('graficos_densidade')

# Plotando um gráfico para cada paciente e salvando-o na pasta
for i in range(patients.shape[1]):
    hiper = patients[:,i]

    f, Pxx = welch(hiper, fs=fs, nperseg=1024)
    psd = Pxx / max(Pxx)

    plt.semilogy(f, psd)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Densidade espectral')
    plt.title('Paciente{}'.format(i+1))   
    plt.savefig('graficos_densidade/paciente{}.png'.format(i+1))
    plt.close()