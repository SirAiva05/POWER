import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
from functions.predict_glucose import predict_glucose
from functions.load_data import load_data
import numpy as np
import matplotlib.pyplot as plt


# Defina a frequência de amostragem (amostras por unidade de tempo)
fs = 0.2 # 1 / 5 amostra por minuto

# Defina a frequência de corte do filtro de Butterworth (Hz)
fc = 0.0016 # 1 / 300 Hz

# Defina a ordem do filtro de Butterworth
order = 3

# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

for dataset in range(0,1):
    df, df_test = load_data(rootdir, 9)
    print("Tamanho do dataset: ", len(df_test))

    # Crie um array com os dados de predição dos níveis de glicose
    glicose = np.array(df_test['Sensor Glucose (mg/dL)'])  # substitua os pontos suspensos pelos dados reais

    # Calcule os coeficientes do filtro de Butterworth
    b, a = butter(order, 2*fc/fs, btype='lowpass', analog=False)

    # Aplique o filtro de Butterworth na série temporal de níveis de glicose
    glicose_filtrada = filtfilt(b, a, glicose)

    # Plote a série temporal de níveis de glicose original e filtrada
    plt.plot(glicose, label='Original')
    plt.plot(glicose_filtrada, label='Filtrada')
    plt.legend()
    plt.show()


