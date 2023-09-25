import pywt
import numpy as np
import matplotlib.pyplot as plt
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


# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

# Defina o tipo de wavelet e o número de escalas desejadas
wavelet = 'db4'
num_scales = 4

threshold = 10

def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Por exemplo, para aplicar o filtro na escala 3:
scale_to_filter = 300

for dataset in range(0,1):
    df, df_test = load_data(rootdir, 9)
    print("Tamanho do dataset: ", len(df_test))

    glucose = np.array(df_test['Sensor Glucose (mg/dL)'])  # substitua os pontos suspensos pelos dados reais

    # Realize a decomposição da série em diferentes escalas usando a transformada wavelet
    coefficients = pywt.wavedec(glucose, wavelet, level=num_scales)

    # Aplique o filtro em uma ou mais escalas selecionadas
    filtered_coeffs = [soft_thresholding(c, threshold) for c in coefficients[1:]]
    #coefficients[scale_to_filter] = pywt.threshold(coefficients[scale_to_filter], 2, mode='soft')

    # Reconstrua a série filtrada usando a transformada wavelet inversa
    #filtered_glicose = pywt.waverec(coefficients, wavelet)
    filtered_glicose = pywt.waverec([coefficients[0]] + filtered_coeffs, wavelet)


    # Plote a série original e a série filtrada
    plt.plot(glucose, label='Original')
    plt.plot(filtered_glicose, label='Filtrada')
    plt.legend()
    plt.show()
