from functions.save_results import save_results
from functions.insert_missing import insert_missing
from functions.evaluate_model import evaluate_model
from functions.data_prep import data_prep
from tensorflow.keras.layers import LSTM, Dense, Flatten, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# def train_model(method, n, p, n_features, method_missing, dayx, activationfcn, activationfcn1, units, units1, units2):

# method = tipo de modelo a desenvolver (string)
# n = tamanho do input do modelo (int)
# p = horizonte de predicao (int)
# dayx = modalidade de predicao (boolean):
#   dayx = True  -> predição do ultimo dia do horizonte de predicao p
#   dayx = False -> predição de todos os dias do horizonte de predicao p
def train_model(method, n, p, dayx): 

    # PARAMS

    n_features = 1

    method_missing = "MEAN"

    optimizerfcn = 'adam'
    lossfcn = 'mse' 

    units1 = 1
    units2 = 1

    if method == 'LSTM': # parametros lstm
        units = 50
        activationfcn =  'relu' 
        activationfcn1 = 'relu' 
    elif method == 'JNN': # parametros jnn
        units = 4
        activationfcn =  'sigmoid'
        activationfcn1 = 'linear'

    try:

        data = pd.read_csv('dataset/myHeartBP.csv', header=None).values
        data = insert_missing(data, n, n_features, method_missing)

        if any(type(x) is str for x in data) or len(np.where(data < 50)[0]) or len(np.where(data > 200)[0]) or np.isnan(data).any():
            raise Exception

    except Exception:
        print('Invalid data input.')
        sys.exit(1)

    else:

        patients = data  # todos os pacientes

        results = []

        # guardar os resultados da avaliacao da performance para cada paciente de teste
        mape_score = []
        mse_score = []
        rmse_score = []
        mae_score = []


        patient_idx = [*range(0, np.size(data, 1), 1)]

        # iteracao de cada paciente para uso como paciente de teste (leave-one-out)
        for test_index in patient_idx:

            train_index = patient_idx.copy()
            train_index.remove(test_index) # remover o indice do paciente de teste do indice de todos os pacientes

            # divisao em dataset de treino e teste
            train_patients, test_patient = patients[:,
                                                    train_index], patients[:, test_index]

            if method == 'LSTM': # criacao modelo LSTM
                model = Sequential()
               
                # 1a hidden layer
                model.add(LSTM(units=50, activation='relu', return_sequences=True,
                               input_shape=(n, n_features)))
                
                # 2a hidden layer
                model.add(LSTM(units=1, activation='relu'))
                
                # output layer
                model.add(Dense(units=1))

                model.compile(optimizer=optimizerfcn, loss=lossfcn)

            elif method == 'JNN': # criacao modelo jump neural network
                # input layer
                inputs = Input(shape=(n, n_features))
                inputs = Flatten()(inputs)

                # passar o input para a hidden layer - parte nao linear
                dense = Dense(units=4, activation='sigmoid')
                hidden = Flatten()(dense(inputs))
                
                # obter o 1 output (o da camada de input) e juntar ao 2 output (o primeiro output esta a saltar a hidden layer) - parte linear
                jump = Concatenate(axis=1)([inputs, hidden])
                outputs = Dense(units=1, activation='linear')(
                    jump) 

                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=optimizerfcn, loss=lossfcn)
                # model.summary()

            elif method == 'LR': 
                model = LinearRegression() # criacao modelo de regressao linear

            for i in range(np.size(train_patients, 1)): # percorrer todos os pacientes de treino

                # separar os dados de cada paciente em arrays de tamanho n para input (X), a cada qual estará associado o output real (y) 
                X, y = data_prep(train_patients[:, i], n, p, dayx) 

                if X.size != 0 and y.size != 0:

                    if method == 'LR':
                        # no caso da regressão linear, o input terá 2 dimenões (array 1 x n)
                        X = X.reshape((X.shape[0], X.shape[1]))
                        model.fit(X, y) 

                    else:
                        # no caso de lstm e jnn, o input terá 3 dimenões (array 1 x n x n_features), para incluir mais features posteriormente
                        # reshape de [padroes, n] para [padroes, n, n_features]
                        X = X.reshape((X.shape[0], X.shape[1], n_features))
                        model.fit(X, y, epochs=300, verbose=0)

            # avaliacao da performance do modelo
            mape, mse, rmse, mae = evaluate_model(model,
                test_patient, n, p, n_features, method, dayx, test_index)

            if not np.isnan(mape) and not np.isnan(mse) and not np.isnan(mae):
                mape_score.append(mape)
                mse_score.append(mse)
                rmse_score.append(rmse)
                mae_score.append(mae)

        # media da avaliacao da performance para todos os pacientes de teste
        result_mape = np.mean(np.absolute(mape_score))
        result_mse = np.mean(np.absolute(mse_score))
        result_rmse = np.mean(np.absolute(rmse_score))
        result_mae = np.mean(np.absolute(mae_score))

        # guardar resultados em csv
        results = save_results(results, method, n, p, result_mape, result_mse, result_rmse, result_mae,
                               activationfcn, activationfcn1, optimizerfcn, lossfcn, dayx)

        header = ['method', 'N', 'P', 'MAPE', 'MSE', 'RMSE',
                  'MAE', 'activationfcn', 'activationfcn1', 'optimizerfcn', 'lossfcn', 'dayx']

        if results.ndim == 1:
            pd.DataFrame(results).transpose().to_csv(
                "results.csv", index=False, header=header)
        else:
            pd.DataFrame(results).to_csv(
                "results.csv", index=False, header=header)


# params
method = "JNN"  # "JNN" # "LSTM" "LR"
n = 10  # 6 # 8 # 10
p = 3  # 5 # 7 # 9
dayx = True  # previsão de apenas do dia P (True) ou dos dias 0:P (False)


train_model('JNN', 8, 3, False)
train_model('JNN', 8, 5, False)
train_model('JNN', 10, 7, False)
train_model('JNN', 8, 9, False)