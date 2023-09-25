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
import pywt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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

def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Por exemplo, para aplicar o filtro na escala 3:
scale_to_filter = 300


# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

for dataset in range(0,1):
    df, df_test = load_data(rootdir, 9)
    print("Tamanho do dataset: ", len(df_test))

    glicose = np.array(df_test['Sensor Glucose (mg/dL)'])  # substitua os pontos suspensos pelos dados reais

    # Calcule os coeficientes do filtro de Butterworth
    b, a = butter(order, 2*fc/fs, btype='lowpass', analog=False)

    # Aplique o filtro de Butterworth na série temporal de níveis de glicose
    glicose_filtrada = filtfilt(b, a, glicose)



    # Realize a decomposição da série em diferentes escalas usando a transformada wavelet
    coefficients = pywt.wavedec(glicose_filtrada, wavelet, level=num_scales)

    # Aplique o filtro em uma ou mais escalas selecionadas
    filtered_coeffs = [soft_thresholding(c, threshold) for c in coefficients[1:]]
    #coefficients[scale_to_filter] = pywt.threshold(coefficients[scale_to_filter], 2, mode='soft')

    # Reconstrua a série filtrada usando a transformada wavelet inversa
    #filtered_glicose = pywt.waverec(coefficients, wavelet)
    filtered_glicose = pywt.waverec([coefficients[0]] + filtered_coeffs, wavelet)
    
    
    # Calcule a derivada da série temporal de glicose
    derivada = np.gradient(filtered_glicose)
    gradiente_significativo = derivada > 0 # vetor booleano indicando valores de gradiente significativos


    # Encontre os índices onde a derivada é positiva e a glicose é acima de 180
    indices_positivos = np.where(derivada > 0)[0]
    momentos_crescentes = [i for i in indices_positivos if filtered_glicose[i] > 180]


    duracao_minima = 24 # duração mínima em minutos
    hiperglicemia_aguda = glicose > 180 # vetor booleano indicando hiperglicemia aguda
    gradiente_acima_do_limiar = derivada > 0 # vetor booleano indicando valores de gradiente acima do limiar
    gradiente_acima_do_limiar_duracao = np.zeros_like(glicose, dtype=bool) # vetor booleano indicando duração do gradiente acima do limiar
    for i in range(len(gradiente_acima_do_limiar)):
        if gradiente_acima_do_limiar[i] and hiperglicemia_aguda[i]:
            gradiente_acima_do_limiar_duracao[i:i+duracao_minima] = True # considera um episódio de duração mínima

    # Crie o gráfico de linha com marcadores
    plt.plot(filtered_glicose)
    plt.plot(momentos_crescentes, filtered_glicose[momentos_crescentes], 'ro')
    plt.xlabel('Tempo')
    plt.ylabel('Glicose')
    plt.title('Série Temporal de Glicose com Marcadores de Momentos Crescentes Acima de 180')
    plt.show()

    # Crie um vetor de tempo para a série temporal de glicose
    tempo = np.arange(len(filtered_glicose))

    # Crie um gráfico com a série temporal de glicose e seus pontos derivados
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Tempo (minutos)')
    ax1.set_ylabel('Glicose', color=color)
    ax1.plot(tempo, filtered_glicose, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Derivada', color=color)
    ax2.plot(tempo, derivada, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(glicose, label='Glicose')
    ax.fill_between(np.arange(len(glicose)), 0, 400, where=hiperglicemia_aguda, alpha=0.3, color='red', label='Hiperglicemia')
    ax.plot(gradiente_significativo * 400, label='Gradiente > 0 mg/dL/min', color='orange')
    ax.plot(gradiente_acima_do_limiar_duracao * 400, label='Gradiente > 0 mg/dL/min por pelo menos 2 h', color='purple')
    ax.set_xlabel('Tempo (minutos)')
    ax.set_ylabel('Glicose (mg/dL)')
    ax.set_title('Série temporal de glicose e gradiente')
    ax.legend()
    plt.show()


    