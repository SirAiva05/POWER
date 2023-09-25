import pywt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
import sys
sys.path.append('..')
from functions.predict_glucose import predict_glucose
from functions.load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import scipy.signal
from scipy.signal import welch, periodogram


# Patient input
rootdir = r'../dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fs = 1/5 # 1 kHz sampling frequency

densidades_espectrais = []
frequencias_totais = []

for dataset in datasets:
    df, df_test = load_data(rootdir, dataset)
    print("Tamanho do dataset: ", len(df_test))

    glucose = np.array(df_test['Sensor Glucose (mg/dL)']) 

    frequencias = rfftfreq(len(glucose), 1/fs) #len patients e numero de amostras por s
    frequencias = frequencias[1:]

    # Criar uma lista vazia para cada densidade espectral
    densidades_espectrais = [[] for _ in frequencias]

    Pxx = rfft(glucose)
    Pxx = Pxx[1:]
    psd = np.abs(Pxx) / max(Pxx)

    for i, p in enumerate(psd):
        densidades_espectrais[i].append(p)
    
densidades_espectrais_media = np.mean(np.stack(densidades_espectrais), axis=0)
frequencias_medias = np.mean(np.stack(frequencias), axis=0)

densidades_espectrais_media = densidades_espectrais_media[:764]


print("Frequências: ", max(frequencias))
print("PSD: ", max(densidades_espectrais_media))

for valor in densidades_espectrais_media:
    if valor == 0:
        break
    valor_anterior = valor

if valor_anterior is not None:
    print("O valor anterior ao primeiro valor igual a zero é:", valor_anterior)
else:
    print("Não há valores diferentes de zero na lista")

plt.semilogy(frequencias, densidades_espectrais_media)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Densidade espectral')
plt.title('Média do espectro de densidade de glicose para todos os pacientes')
plt.show()

