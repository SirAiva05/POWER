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


    ##### FILTRAGEM ##########

    # Crie um array com os dados de predição dos níveis de glicose
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

    '''
    # Plote a série original e a série filtrada
    plt.plot(glicose, label='Original')
    plt.plot(filtered_glicose, label='Filtrada')
    plt.legend()
    plt.show()
    '''

    ###### PREDIÇÃO ########

    # escolha do horizonte de previsao: 2h, 4h, 12h ou None
    pred_horizon = None
    block_size_2 = 2 * 12
    block_size_4 = 4 * 12
    block_size_12 = 12 * 12
    a = 0 
    b = 144
    actual_2h = []
    actual_4h = []
    actual_12h = []
    predict_2h = []
    predict_4h = []
    predict_12h = []
    patient_input = filtered_glicose[a:b]
    result, error_code = predict_glucose(patient_input, pred_horizon)
    predict_2h.append(result[0])
    predict_4h.append(result[1])
    #predict_12h.append(result[2])
    actual_2h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_2-1])
    actual_4h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_4-1])
    #actual_12h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_12-1])

    for i in range(0, len(df_test) - block_size_4 - 1):
        a += 1
        b += 1
        if b >= len(df_test) - block_size_4 - 1:
            break
        patient_input = filtered_glicose[a:b]
        result, error_code = predict_glucose(patient_input, pred_horizon)
        predict_2h.append(result[0])
        predict_4h.append(result[1])
        #predict_12h.append(result[2])
        actual_2h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_2-1])
        actual_4h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_4-1])
        #actual_12h.append(df_test['Sensor Glucose (mg/dL)'][b + block_size_12-1])


    medias_actuais_2h = []
    medias_predict_2h = []

    medias_actuais_12h = []
    medias_predict_12h = []

    medias_actuais_4h = []
    medias_predict_4h = []

    # Iterando sobre todos os elementos da lista
    for i in range(len(predict_2h) - block_size_2 + 1):
        
        # Selecionando o grupo de elementos

        grupo_2h_a = df_test['Sensor Glucose (mg/dL)'][i:i+block_size_2]
        grupo_2h_p = predict_2h[i:i+block_size_2]

        grupo_4h_a = df_test['Sensor Glucose (mg/dL)'][i:i+block_size_2]
        grupo_4h_p = predict_4h[i:i+block_size_2]
        
        #grupo_12h_a = df_test['Sensor Glucose (mg/dL)'][i:i+block_size]
        #grupo_12h_p = predict_12h[i:i+block_size]

        # Calculando a média do grupo
        media_a_2 = sum(grupo_2h_a) / len(grupo_2h_a)
        media_p_2 = sum(grupo_2h_p) / len(grupo_2h_p)

        media_a_4 = sum(grupo_4h_a) / len(grupo_4h_a)
        media_p_4 = sum(grupo_4h_p) / len(grupo_4h_p)

        #media_a_12 = sum(grupo_12h_a) / len(grupo_12h_a)
        #media_p_12 = sum(grupo_12h_p) / len(grupo_12h_p)
        
        # Adicionando a média à lista de médias
        medias_actuais_2h.append(media_a_2)
        medias_predict_2h.append(media_p_2)

        medias_actuais_4h.append(media_a_4)
        medias_predict_4h.append(media_p_4)

        #medias_actuais_12h.append(media_a_12)
        #medias_predict_12h.append(media_p_12)
    
    block_size = 144
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.axhline(y=150, color='g', linewidth=2)
    plt.axhline(y=170, color='y', linewidth=2)
    plt.axhline(y=180, color='r', linewidth=2)
    plt.plot(df_test[0 : 1000].index, medias_actuais_2h[0 : 1000], label="Actual")
    plt.plot(df_test[block_size : 1000].index, predict_2h[0 : 1000-block_size], label="Predict")
    # towards right
    plt.xticks(rotation=30, ha='right')
    
    # Providing x and y label to the chart
    plt.xlabel('Time (5 minutos intervalo)')
    plt.ylabel('Sensor Glucose (mg/dL) [2h]')
    plt.legend()

    plt.subplot(1, 2, 2) # index 2
    plt.axhline(y=150, color='g', linewidth=2)
    plt.axhline(y=170, color='y', linewidth=2)
    plt.axhline(y=180, color='r', linewidth=2)
    plt.plot(df_test[0 : 1000].index, medias_actuais_4h[0 : 1000], label="Actual")
    plt.plot(df_test[block_size : 1000].index, predict_4h[0 : 1000-block_size], label="Predict")
    # towards right
    plt.xticks(rotation=30, ha='right')
    
    # Providing x and y label to the chart
    plt.xlabel('Time (5 minutos intervalo)')
    plt.ylabel('Sensor Glucose (mg/dL) [4h]')
    plt.legend()

    plt.show()


