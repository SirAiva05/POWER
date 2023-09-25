from functions.predict_glucose import predict_glucose
from functions.load_data import load_data
from functions.train_model import train_glucose
import numpy as np

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back :])
        Y.append(dataset[i :])
    return X, Y

# Patient input
rootdir = r'dataset/1_trial'

datasets_train = [0, 2, 3, 4, 6, 7, 8]
datasets_test = [1, 5, 9]

look_back = 24

for dataset in datasets_train:
    df, df_test = load_data(rootdir, dataset)
    print("Tamanho do dataset: ", len(df_test))


    train_X, train_Y = create_dataset(df_test['Sensor Glucose (mg/dL)'], look_back)
    print(len(train_X), len(train_Y))


    # escolha do horizonte de previsao: 2h, 4h, 12h ou None
    pred_horizon = 2

    result, error_code = train_glucose(train_X, train_Y, pred_horizon)




    
