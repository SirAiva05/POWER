import numpy as np
import pickle
import sys

# predicao do array de input usando o modelo treinado
def BP_pred(model, test_patient_input, n, pred_horizon, n_features, method, dayx):

    try:

        input = np.array(test_patient_input)
        temp_input = list(input)

        if any(type(x) is str for x in temp_input) or any(x < 50 for x in temp_input) or any(x > 200 for x in temp_input) or np.isnan(temp_input).any():
            raise Exception

        # model = pickle.load(open('model.sav', 'rb'))

    except Exception:
        print('Error.')
        sys.exit(1)

    else:

        if not dayx: # predicao de varios (pred_horizon) dias

            # array com os outputs gerados
            lst_output = []
            i = 0

            while (i < pred_horizon): # predicao de pred_horizon valores
                if(len(temp_input) > n):

                    if method == 'LSTM' or method == 'JNN':
                        input = np.array(temp_input[1:]).reshape(
                            (1, n, n_features))
                        yhat = model.predict(input)
                        temp_input.append(yhat[0][0]) # adicionar o valor previsto ao final do array de input
                        temp_input = temp_input[1:] # descartar o primeiro valor
                        lst_output.append(yhat[0][0]) # adicionar valor previsto no array da predicao

                    elif method == 'LR':
                        input = np.array(temp_input[1:]).reshape((1, n))
                        yhat = model.predict(input)
                        temp_input.append(yhat[0]) # adicionar o valor previsto ao final do array de input
                        temp_input = temp_input[1:] # descartar o primeiro valor
                        lst_output.append(yhat[0]) # adicionar valor previsto no array da predicao

                else:

                    if method == 'LSTM' or method == 'JNN':
                        input = input.reshape((1, n, n_features))
                        yhat = model.predict(input)
                        temp_input.append(yhat[0][0])
                        lst_output.append(yhat[0][0])

                    elif method == 'LR':
                        input = input.reshape((1, n))
                        yhat = model.predict(input)
                        temp_input.append(yhat[0])
                        lst_output.append(yhat[0])

                i += 1

        elif dayx: # predicao de um unico dia
            lst_output = 0 
            if method == 'LSTM' or method == 'JNN':
                input = np.array(temp_input).reshape((1, n, n_features))
                yhat = model.predict(input)
                lst_output = yhat[0][0] # output gerado

            elif method == 'LR':
                input = np.array(temp_input).reshape((1, n))
                yhat = model.predict(input)
                lst_output = yhat[0] # output gerado

        # print('Real Output: ', test_patient_output)
        # print('Predicted Output: ', lst_output)

        return lst_output
