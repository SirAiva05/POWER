from predict_bp import *
from train.insert_missing import * 
import math
import matplotlib.pyplot as plt
import os

n = [4, 6, 8, 10]
pred_horizon = [1, 3, 7]

n_features = 1
method_missing = "MEAN"

data = pd.read_csv('dataset/myHeartBP.csv', header=None).values
data = insert_missing(data, n, n_features, method_missing)
patients = data  # todos os paciente

# Definindo o valor da linha de hipertensão alta
linha_alta = 140

# Criando a pasta para salvar os gráficos
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# Plotando um gráfico para cada paciente e salvando-o na pasta
for i in range(patients.shape[1]):
    plt.plot(patients[:,i])
    plt.axhline(y=linha_alta, color='r', linestyle='--')
    plt.title('Paciente {}'.format(i+1))
    plt.xlabel('Dia')
    plt.ylabel('Nível de hipertensão')
    plt.savefig('graficos/paciente{}.png'.format(i+1))
    plt.close()