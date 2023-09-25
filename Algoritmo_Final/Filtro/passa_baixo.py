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

for dataset in datasets_train:
    df, df_test = load_data(rootdir, dataset)
    print("Tamanho do dataset: ", len(df_test))

    # Crie um array com os dados de predição dos níveis de glicose
    glicose = np.array(df_test['Sensor Glucose (mg/dL)'])  # substitua os pontos suspensos pelos dados reais

    # Defina uma lista de valores possíveis para window_length e polyorder
    window_lengths = range(3, 25 ,2)
    polyorders = range(2, 6)

    # Inicialize uma variável para armazenar o menor MSE encontrado até agora
    menor_mse = float('inf')

    # Inicialize variáveis para armazenar o melhor window_length e polyorder encontrados até agora
    melhor_window_length = None
    melhor_polyorder = None

    # Loop sobre todas as combinações de window_length e polyorder
    for window_length in window_lengths:
        for polyorder in polyorders:
            if polyorder < window_length:
                print(window_length, polyorder)
                # Aplique o filtro passa baixo de Savitzky-Golay
                glicose_filtrada = savgol_filter(glicose, window_length=window_length, polyorder=polyorder, mode='nearest')

                            # Plote o sinal original e o sinal filtrado com os melhores parâmetros
                plt.plot(glicose, label='Original')
                plt.plot(glicose_filtrada, label='Filtrado')
                plt.legend()
                plt.show()

                print(f'Melhor window_length: {melhor_window_length}')
                print(f'Melhor polyorder: {melhor_polyorder}')
                print(f'Menor MSE: {menor_mse} \n')

                # Calcule o erro quadrático médio (MSE) entre o sinal original e o sinal filtrado
                mse = ((glicose - glicose_filtrada)**2).mean()
                # Se o MSE atual for menor que o menor MSE encontrado até agora,
                # atualize as variáveis de melhor escolha
                if mse < menor_mse:
                    menor_mse = mse
                    melhor_window_length = window_length
                    melhor_polyorder = polyorder

    # Aplique o filtro passa baixo de Savitzky-Golay com os melhores parâmetros encontrados
    glicose_filtrada = savgol_filter(glicose, window_length=melhor_window_length, polyorder=melhor_polyorder, mode='nearest')

    



