from functions.predict_glucose import predict_glucose
from functions.load_data import load_data
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

######      METRICAS     ###### 

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Calculate mean absolute error
def mae_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
	return sum_error / float(len(actual))

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Define the function to return the MAPE values
def mape(actual, predicted):
    APE = []
    for i in range(len(actual)):
        per_err = (actual[i] - predicted[i])/actual[i]
        per_err = abs(per_err)
        APE.append(per_err)

    MAPE = (sum(APE)/len(APE))*100
    return MAPE

##################################################

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

# Patient input
rootdir = r'dataset/1_trial'
for n in datasets_test:
    df, df_test = load_data(rootdir, 9)
    print("Tamanho do dataset: ", len(df_test))
 
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
    patient_input = df_test['Sensor Glucose (mg/dL)'][a:b]
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
        patient_input = df_test['Sensor Glucose (mg/dL)'][a:b]
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
            
    accuracy_2h = accuracy_metric(actual_2h, predict_2h)
    accuracy_4h = accuracy_metric(actual_4h, predict_4h)
    #accuracy_12h = accuracy_metric(actual_12h, predict_12h)
    print("ACC 2h --> ", accuracy_2h)
    print("ACC 4h --> ", accuracy_4h)
    #print("ACC 12h --> ", accuracy_12h)

    mae_2h = mae_metric(actual_2h, predict_2h)
    mae_4h = mae_metric(actual_4h, predict_4h)
    #mae_12h = mae_metric(actual_12h, predict_12h)
    print("MAE 2h --> ", mae_2h)
    print("MAE 4h --> ",mae_4h)
    #print("MAE 12h --> ",mae_12h)

    rmse_2h = rmse_metric(actual_2h, predict_2h)
    rmse_4h = rmse_metric(actual_4h, predict_4h)
    #rmse_12h = rmse_metric(actual_12h, predict_12h)
    print("RMSE 2h --> ",rmse_2h)
    print("RMSE 4h --> ",rmse_4h)
    #print("RMSE 12h --> ",rmse_12h)


    mape_2h = mape(actual_2h, predict_2h)
    mape_4h = mape(actual_4h, predict_4h)
    #mape_12h = mape(actual_12h, predict_12h)
    print("MAPE 2h --> ",mape_2h)
    print("MAPE 4h --> ",mape_4h)
    #print("MAPE 12h --> ",mape_12h)

    predict_2h.clear()
    predict_4h.clear()
    predict_12h.clear()
    actual_2h.clear()
    actual_4h.clear()
    actual_12h.clear()