import numpy as np
import math
from functions.BP_pred import BP_pred
from functions.plot_predict import plot_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 
def evaluate_model(model, test_patient, n, pred_horizon, n_features, method, dayx, test_index):

    pae_score = [] # percent absolute score
    mse_score = []
    rmse_score = []
    mae_score = []

    test_patient_prediction = []
    

    for idx in range(0, len(test_patient)-n-pred_horizon+1):

        test_patient_input = test_patient[idx:idx+n] # input para predicao
        if dayx:
            test_patient_output = test_patient[idx+n+pred_horizon-1] # output real
        else:
            test_patient_output = test_patient[idx+n:idx+n+pred_horizon] # outputs reais

        # corre os dados do paciente de teste numa janela de intervalo n
        # test_patient_input é o input do modelo e test_patient_output é o valor real q será comparado com o output do modelo

        lst_output = BP_pred(model, test_patient_input, n,
                             pred_horizon, n_features, method, dayx)

        #percent absolute error
        pae = np.mean(
            np.abs((test_patient_output - lst_output) / test_patient_output)) * 100

        if dayx:
            mse = (lst_output - test_patient_output)**2
            mae = np.abs(lst_output - test_patient_output)
        else:
            mse = mean_squared_error(test_patient_output, lst_output)
            mae = mean_absolute_error(test_patient_output, lst_output)

        rmse = math.sqrt(mse)

        if not np.isnan(pae) and not np.isnan(mse) and not np.isnan(mae):
            pae_score.append(pae)
            mse_score.append(mse)
            rmse_score.append(rmse)
            mae_score.append(mae)


        if dayx:
            test_patient_prediction.append(lst_output)
        else:
            plot_predict(test_patient_output, lst_output,
                 test_index, idx, method, n, pred_horizon, dayx)
            test_patient_prediction.append(lst_output[-1])

        

    patient_error_pae = np.mean(np.absolute(pae_score))
    patient_error_mse = np.mean(np.absolute(mse_score))
    patient_error_rmse = np.mean(np.absolute(rmse_score))
    patient_error_mae = np.mean(np.absolute(mae_score))

    if dayx:
        plot_predict(test_patient, test_patient_prediction,
                    test_index, patient_error_pae, method, n, pred_horizon, dayx)

    return patient_error_pae, patient_error_mse, patient_error_rmse, patient_error_mae
