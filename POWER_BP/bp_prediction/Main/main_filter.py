from predict_bp import *
from train.insert_missing import * 
import math
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
import pywt
import os

# Criando a pasta para salvar os gráficos
if not os.path.exists('graficos_filter/graficos_N4_P3'):
    os.makedirs('graficos_filter/graficos_N4_P3')

############################################################################################################

def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Por exemplo, para aplicar o filtro na escala 3:
scale_to_filter = 300

# Defina a ordem do filtro de Butterworth
order = 3

# Defina a frequência de amostragem (amostras por unidade de tempo)
fs = 1

# Defina a frequência de corte do filtro de Butterworth (Hz)
fc = 0.35

# Defina o tipo de wavelet e o número de escalas desejadas
wavelet = 'db4'
num_scales = 4

threshold = 10

############################################################################################################
# Definindo o valor da linha de hipertensão alta
linha_alta = 140

# n = [4, 6, 8, 10]
# pred_horizon = [1, 3, 7]

n = [4]
pred_horizon = [3]

n_features = 1
method_missing = "MEAN"

data = pd.read_csv('dataset/myHeartBP.csv', header=None).values
data = insert_missing(data, n, n_features, method_missing)
patients = data  # todos os paciente

header = ['N', 'P', 'MAPE', 'MSE', 'RMSE', 'MAE']

pre=[]
act=[]

for dim in n:
    print("------------","INPUT -->", dim, "------------")

    results = []
    # guardar os resultados da avaliacao da performance para cada paciente de teste
    
    pae_score_1 = []
    mape_score_1 = []
    mse_score_1 = []
    rmse_score_1 = []
    mae_score_1 = []
    test_patient_prediction_1 = []

    pae_score_3 = []
    mape_score_3 = []
    mse_score_3 = []
    rmse_score_3 = []
    mae_score_3 = []
    test_patient_prediction_3 = []

    pae_score_7 = []
    mape_score_7 = []
    mse_score_7 = []
    rmse_score_7 = []
    mae_score_7 = []
    test_patient_prediction_7 = []


    patient_idx = [*range(0, np.size(data, 1), 1)]
    last_item = patient_idx[-1]

    # iteracao de cada paciente para uso como paciente de teste (leave-one-out)
    for test_index in patient_idx:
        print("------------","PACIENTE -->", test_index+1, "------------")

        train_index = patient_idx.copy()
        train_index.remove(test_index) # remover o indice do paciente de teste do indice de todos os paciente
        # divisao em dataset de treino e teste
        train_patients, test_patient = patients[:,train_index], patients[:, test_index]
        ############################################################################################################
        
        # Calcule os coeficientes do filtro de Butterworth
        b, a = butter(order, fc, btype='lowpass', analog=False, fs=fs )

        # Aplique o filtro de Butterworth na série temporal de níveis de glicose
        filtered_hiper = filtfilt(b, a, test_patient)

        # Realize a decomposição da série em diferentes escalas usando a transformada wavelet
        # coefficients = pywt.wavedec(hiper_filtrada, wavelet, level=num_scales)

        # # Aplique o filtro em uma ou mais escalas selecionadas
        # filtered_coeffs = [soft_thresholding(c, threshold) for c in coefficients[1:]]
        # #coefficients[scale_to_filter] = pywt.threshold(coefficients[scale_to_filter], 2, mode='soft')

        # # Reconstrua a série filtrada usando a transformada wavelet inversa
        # #filtered_glicose = pywt.waverec(coefficients, wavelet)
        # filtered_hiper = pywt.waverec([coefficients[0]] + filtered_coeffs, wavelet)

############################################################################################################

        for horizonte in pred_horizon:
            for idx in range(0, len(filtered_hiper)-dim-horizonte+1):

                test_patient_input = filtered_hiper[idx:idx+dim] # input para predicao
                test_patient_output = filtered_hiper[idx+dim+horizonte-1] # output real
                
                predictions, error_code = predict_bp(test_patient_input, horizonte)

                if horizonte == 1:

                    pre.append(predictions[0])
                    act.append(test_patient_output)
                    pae_1 = np.mean(np.abs((test_patient_output - predictions[0]) / test_patient_output)) * 100
                    mse_1 = (predictions[0] - test_patient_output)**2
                    mae_1 = np.abs(predictions[0] - test_patient_output)
                    rmse_1 = math.sqrt(mse_1)

                    mape_score_1.append(pae_1)
                    mse_score_1.append(mse_1)
                    rmse_score_1.append(rmse_1)
                    mae_score_1.append(mae_1)

                    test_patient_prediction_1.append(predictions[0])

                elif horizonte == 3:
                    pre.append(predictions[1])
                    act.append(test_patient_output)
                    pae_3 = np.mean(np.abs((test_patient_output - predictions[1]) / test_patient_output)) * 100
                    mse_3 = (predictions[1] - test_patient_output)**2
                    mae_3 = np.abs(predictions[1] - test_patient_output)
                    rmse_3 = math.sqrt(mse_3)

                    mape_score_3.append(pae_3)
                    mse_score_3.append(mse_3)
                    rmse_score_3.append(rmse_3)
                    mae_score_3.append(mae_3)

                    test_patient_prediction_3.append(predictions[1])


                elif horizonte == 7:
                    pre.append(predictions[2])
                    act.append(test_patient_output)
                    pae_7 = np.mean(np.abs((test_patient_output - predictions[2]) / test_patient_output)) * 100
                    mse_7 = (predictions[2] - test_patient_output)**2
                    mae_7 = np.abs(predictions[2] - test_patient_output)
                    rmse_7 = math.sqrt(mse_7)

                    mape_score_7.append(pae_7)
                    mse_score_7.append(mse_7)
                    rmse_score_7.append(rmse_7)
                    mae_score_7.append(mae_7)

                    test_patient_prediction_7.append(predictions[2])

            plt.plot(range(1, len(filtered_hiper)+1), filtered_hiper, label="Actual")
            plt.plot(range(dim+horizonte ,len(filtered_hiper)+1), pre, label="Predict")
            plt.axhline(y=linha_alta, color='r', linestyle='--')
            plt.title('Paciente {}'.format(test_index+1))
            plt.xlabel('Dia')
            plt.ylabel('Nível de hipertensão')
            plt.savefig('graficos_filter/graficos_N4_P3/paciente{}.png'.format(test_index+1))
            plt.close()
            
            pre.clear()

    print("------------","HORIZONTE:", 1, "------------")
    result_mape_1 = np.mean(np.absolute(mape_score_1))
    result_mse_1 = np.mean(np.absolute(mse_score_1))
    result_rmse_1 = np.mean(np.absolute(rmse_score_1))
    result_mae_1 = np.mean(np.absolute(mae_score_1))
    arr_1 = np.array([dim, 1, result_mape_1, result_mse_1, result_rmse_1, result_mae_1])
    print(arr_1)
    
    print("------------","HORIZONTE:", 3, "------------")
    result_mape_3 = np.mean(np.absolute(mape_score_3))
    result_mse_3 = np.mean(np.absolute(mse_score_3))
    result_rmse_3 = np.mean(np.absolute(rmse_score_3))
    result_mae_3 = np.mean(np.absolute(mae_score_3))
    arr_3 = np.array([dim, 3, result_mape_3, result_mse_3, result_rmse_3, result_mae_3])
    print(arr_3)
    
    print("------------","HORIZONTE:", 7, "------------")
    result_mape_7 = np.mean(np.absolute(mape_score_7))
    result_mse_7 = np.mean(np.absolute(mse_score_7))
    result_rmse_7 = np.mean(np.absolute(rmse_score_7))
    result_mae_7 = np.mean(np.absolute(mae_score_7))
    arr_7 = np.array([dim, 7, result_mape_7, result_mse_7, result_rmse_7, result_mae_7])
    print(arr_7)

    lines = [arr_1, arr_3, arr_7]


