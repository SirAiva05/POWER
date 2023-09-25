import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity
tf.compat.v1.logging.ERROR
warnings.filterwarnings('ignore', module='tensorflow')

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from functions.trend_data import trend_data
from functions.load_model import load_model
from functions.validate_glucose_data import validate_glucose_data

def predict_glucose(patient_input, pred_horizon):
    error_code, patient_input = validate_glucose_data(patient_input, pred_horizon)
    
    result = [-1, -1, -1]
    
    if error_code == 0:
    
        size_block = 24 #os dados sao divididos em blocos de 2h
        
        result = [0, 0, 0] # vetor com o resultado: 1 posicao previsao 2h,
                                                  # 2 posicao previsao 4h,
                                                  # 3 posicao previsao 12h
        
        if pred_horizon == 2: 
            if len(patient_input) >= 24 and len(patient_input) < 48:
                patient_input = patient_input[-24:]
                lb = 2
            elif len(patient_input) >= 48 and len(patient_input) < 72:
                patient_input = patient_input[-48:]
                lb = 4
            elif len(patient_input) >= 72 and len(patient_input) < 96:
                patient_input = patient_input[-72:]
                lb = 6
            elif len(patient_input) >= 96 and len(patient_input) < 144:
                patient_input = patient_input[-96:]
                lb = 8
            elif len(patient_input) >= 144:
                patient_input = patient_input[-144:]
                lb = 12
                
            ph = int(pred_horizon/(size_block*5/60)) #numero de blocos a prever
            #obter as tendencias dos dados
            data_trend = trend_data(patient_input, size_block)   
                
            #normalizar dados
            scaler = MinMaxScaler(feature_range=(- 1, 1))
            train_data = scaler.fit_transform(data_trend.reshape(-1,1))
            
            loaded_model = load_model(pred_horizon, lb) #Load correct model
            
            
            pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
            pred = np.round(scaler.inverse_transform(pred))

            
            result[0] = float(pred[-1])
            
        elif pred_horizon == 4: 
            if len(patient_input) >= 48 and len(patient_input) < 96:
                patient_input = patient_input[-48:]
                lb = 4
            elif len(patient_input) >= 96 and len(patient_input) < 144:
                patient_input = patient_input[-96:]
                lb = 8
            elif len(patient_input) >= 144:
                patient_input = patient_input[-144:]
                lb = 12
            
            ph = int(pred_horizon/(size_block*5/60)) #numero de blocos a prever
            
            #obter as tendencias dos dados
            data_trend = trend_data(patient_input, size_block)   
                
            #normalizar dados
            scaler = MinMaxScaler(feature_range=(- 1, 1))
            train_data = scaler.fit_transform(data_trend.reshape(-1,1))
            
            loaded_model = load_model(pred_horizon, lb) #Load correct model
            pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
            pred = np.round(scaler.inverse_transform(pred))
            
            result[1] = float(pred[-1])
            
        elif pred_horizon == 12: 
            if len(patient_input) >= 144 and len(patient_input) < 288:
               patient_input = patient_input[-144:]
               lb = 12
            elif len(patient_input) >= 288:
               patient_input = patient_input[-288:]
               lb = 24
            
            ph = int(pred_horizon/(size_block*5/60)) #numero de blocos a prever
            
            #obter as tendencias dos dados
            data_trend = trend_data(patient_input, size_block)   
                
            #normalizar dados
            scaler = MinMaxScaler(feature_range=(- 1, 1))
            train_data = scaler.fit_transform(data_trend.reshape(-1,1))
            
            loaded_model = load_model(pred_horizon, lb) #Load correct model
            pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
            pred = np.round(scaler.inverse_transform(pred))
            
            result[2] = float(pred[-1])
        
        elif pred_horizon == None:
            
            if len(patient_input) >= 24 and len(patient_input) < 48:
                patient_input= patient_input[-24:]
                horizont = 2
                lb = 2
                
                ph = int(horizont/(size_block*5/60)) #numero de blocos a prever
                
                #obter as tendencias dos dados
                data_trend = trend_data(patient_input, size_block)   
                    
                #normalizar dados
                scaler = MinMaxScaler(feature_range=(- 1, 1))
                train_data = scaler.fit_transform(data_trend.reshape(-1,1))
                
                loaded_model = load_model(horizont, lb) #Load correct model
                pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
                pred = np.round(scaler.inverse_transform(pred))
                    
                result[0] = float(pred[-1])
            
            if len(patient_input) >= 48 and len(patient_input) < 144:
                
                if len(patient_input) >= 48 and len(patient_input) < 72:
                    patient_input_2 = patient_input[-48:]
                    lb_2 = 4
                elif len(patient_input) >= 72 and len(patient_input) < 96:
                    patient_input_2 = patient_input[-72:]
                    lb_2 = 6
                elif len(patient_input) >= 96 and len(patient_input) < 144:
                    patient_input_2 = patient_input[-96:]
                    lb_2 = 8
                            
                ph = int(2/(size_block*5/60)) #numero de blocos a prever
                
                #obter as tendencias dos dados
                data_trend = trend_data(patient_input_2, size_block)
                    
                #normalizar dados
                scaler = MinMaxScaler(feature_range=(- 1, 1))
                train_data = scaler.fit_transform(data_trend.reshape(-1,1))
                
                loaded_model = load_model(2, lb_2) #Load correct model
                pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
                pred = np.round(scaler.inverse_transform(pred))
                
                result[0] = float(pred[-1])
            
                if len(patient_input) >= 48 and len(patient_input) < 96:
                    patient_input_4 = patient_input[-48:]
                    lb_4 = 4
                elif len(patient_input) >= 96 and len(patient_input) < 144:
                    patient_input_4 = patient_input[-96:]
                    lb_4 = 8
                
                ph = int(4/(size_block*5/60)) #numero de blocos a prever
                
                #obter as tendencias dos dados
                data_trend = trend_data(patient_input_4, size_block)   
                    
                #normalizar dados
                scaler = MinMaxScaler(feature_range=(- 1, 1))
                train_data = scaler.fit_transform(data_trend.reshape(-1,1))
                
                loaded_model = load_model(4, lb_4) #Load correct model
                pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
                pred = np.round(scaler.inverse_transform(pred))
                
                result[1] = float(pred[-1])
            
            if len(patient_input) >= 144:  
                
                horizont = [2, 4, 12]
                
                patient_input_12 = patient_input[-144:]
    
                for i in range(0,3):
                    
                    ph = int(horizont[i]/(size_block*5/60)) #numero de blocos a prever
                    
                    #obter as tendencias dos dados
                    data_trend = trend_data(patient_input_12, size_block)   
                        
                    #normalizar dados
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    train_data = scaler.fit_transform(data_trend.reshape(-1,1))
                    
                    loaded_model = load_model(horizont[i], 12) #Load correct model
                    pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
                    pred = np.round(scaler.inverse_transform(pred))
                    
                    result[i] = float(pred[-1])
            
                if len(patient_input) >= 288:
                    
                    patient_input = patient_input[-288:]
                    pred_horizon = 12
                    lb = 24
                            
                    ph = int(pred_horizon/(size_block*5/60)) #numero de blocos a prever
                    
                    #obter as tendencias dos dados
                    data_trend = trend_data(patient_input, size_block)   
                        
                    #normalizar dados
                    scaler = MinMaxScaler(feature_range=(- 1, 1))
                    train_data = scaler.fit_transform(data_trend.reshape(-1,1))
                    
                    loaded_model = load_model(pred_horizon, lb) #Load correct model
                    pred = loaded_model.predict(train_data[-ph:], verbose=0) #Make prediction
                    pred = np.round(scaler.inverse_transform(pred))
                    
                    result[2] = float(pred[-1])

    return result, error_code

