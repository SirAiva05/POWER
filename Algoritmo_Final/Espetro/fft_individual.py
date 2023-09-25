from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
from functions.predict_glucose import predict_glucose
from functions.load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import scipy.signal
from scipy.signal import welch, periodogram
import os



# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fs = 1/300 # 1 kHz sampling frequency

# Criando a pasta para salvar os gráficos
if not os.path.exists('graficos'):
    os.makedirs('graficos')


for dataset in datasets:
    df, df_test = load_data(rootdir, dataset)
    print("Tamanho do dataset: ", len(df_test))

    glucose = np.array(df_test['Sensor Glucose (mg/dL)']) 

    f, Pxx = welch(glucose, fs=fs, nperseg=1024)
    psd = Pxx / max(Pxx)

    plt.semilogy(f, psd)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Densidade espectral')
    plt.title('Paciente' + str(dataset))    
    plt.savefig('graficos/paciente{}.png'.format(str(dataset)))
    plt.close()