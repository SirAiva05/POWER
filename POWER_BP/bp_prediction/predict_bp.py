import numpy as np
from validate_bp_data import validate_bp_data
from tensorflow import keras
import pandas as pd

# predicao usando o modelo treinado com os dados de todos os pacientes para o respetivo tamanho de input e horizonte de predicao
# patient_input = array de tamanho entre 3 e 10 dias com os valores de BP observados do paciente (entre 50mmHg e 200mmHg)
# pred_horizon = horizonte de predicao que se pretende obter: 1, 3, 7 ou None (caso seja None, sao apresentadas as 3 predicoes) 
def predict_bp(patient_input, pred_horizon):

    # PARAMS
    n_features = 1
    input_size = len(patient_input)

    patient_input = pd.DataFrame(patient_input).values

    # validacao dos inputs da funcao
    error_code, patient_input = validate_bp_data(patient_input, pred_horizon)

    # array predefinido (permanecera com -1 caso ocorra algum erro)
    prediction = [-1, -1, -1]

    # inputs validos
    if not error_code:

        input = np.array(patient_input)
        temp_input = list(input)
        prediction = [0, 0, 0]

        if pred_horizon == None: # caso seja None, sao apresentadas as 3 predicoes
            horizon_list = [1, 3, 7]
        else: # apresentada apenas a predicao selecionada e as outras serao 0
            horizon_list = [pred_horizon]

        for pred_horizon in horizon_list: 
            model_path = 'models/' + str(input_size) + \
                '_' + str(pred_horizon)
            model = keras.models.load_model(model_path) # importacao do modelo treinado para o tamanho de input e horizonte de predicao selecionados

            input = np.array(temp_input).reshape((1, input_size, n_features))
            yhat = model.predict(input)

            if pred_horizon == 1: # primeira posicao no array prediction
                prediction[0] = yhat[0][0]
            elif pred_horizon == 3: # segunda posicao no array prediction
                prediction[1] = yhat[0][0]
            elif pred_horizon == 7: # terceira posicao no array prediction
                prediction[2] = yhat[0][0]

    return prediction, error_code

