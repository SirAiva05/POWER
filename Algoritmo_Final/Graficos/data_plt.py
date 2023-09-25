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
from scipy.fft import fft, fftfreq, ifft
import scipy.signal
from scipy.signal import welch, periodogram


# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


for dataset in datasets:
    df, df_test = load_data(rootdir, dataset)
    print("Tamanho do dataset: ", len(df_test))

    glucose = np.array(df_test['Sensor Glucose (mg/dL)']) 

    # Defina o eixo X manualmente com intervalos de 5 em 5 minutos
    x_axis = range(0, len(glucose)*5, 5)

    # Divida o eixo X por 60 para transformar os minutos em horas
    x_axis_horas = [x/60 for x in x_axis]

    # Plot a série temporal com o eixo X em horas
    plt.plot(x_axis_horas, glucose)
    plt.title('Paciente' + str(dataset))
    plt.axhline(y=150, color='g', linewidth=2)
    plt.axhline(y=170, color='y', linewidth=2)
    plt.axhline(y=180, color='r', linewidth=2)
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Glicose')
    plt.show()