POWER_BP

- A pasta dataset contem um ficheiro csv com os dados da varíavel de BP presente no dataset do MyHeart, cada coluna corresponde a um paciente (41) e cada linha a uma medição diária (60).

- A pasta bp_model_evaluation contem os códigos de como foram treinados e avaliados os modelos CBR (bp_model_evaluation_cbr) e as redes neuronais LR, JNN e LSTM (bp_model_evaluation_nn).
Abordagem leave-one-out para todos os pacientes do dataset.
Usar funções CBR_predict_bp.py e train_model.py.

- A pasta bp_prediction contem o código para predição utilizando os modelos treinados do modelo selecionado, JNN (predict_bp.py). 
Dentro dela existe uma pasta models com vários modelos treinados com os dados de todos os pacientes disponíveis para predicao consoante diferentes tamanhos de input e horizontes de predição, conforme o estabelecido. 
A função predict_bp.py toma como input uma porção de dados de um paciente (3 a 10 dias de medições diárias) e consoante o horizonte de predição (1, 3 e/ou 7 dias) desejado é feita a mesma, importando os modelos previamente treinados.
Existe também uma pasta train onde podem ser treinados modelos para outros tamanhos de input e horizontes de predição (train_model.py).
Para melhor compreensão do algoritmo final encontra-se o ficheiro power_apis_bp.pdf que explica as funções utilizados e as respetivas variáveis de entrada e saída.

- A pasta trained_models é uma cópia da pasta models existente em bp_prediction, com os modelos treinados com os dados de todos os pacientes disponíveis para 3 a 10 dias de input e 1, 3 e 7 dias de horizonte de predição.
