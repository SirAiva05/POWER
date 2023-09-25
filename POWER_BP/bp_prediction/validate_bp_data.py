import numpy as np
import pandas as pd


def validate_bp_data(patient_input, pred_horizon):

    try:
        error_code = 0

        if not all(isinstance(x[0], (int, float, np.int64)) for x in patient_input) or np.isnan(patient_input).any() or len(patient_input) not in [*range(3, 11)] or not (type(pred_horizon) == int or pred_horizon == None) or pred_horizon not in [1, 3, 7, None]:
            raise Exception
        else:
            patient_input[patient_input > 200] = 200
            patient_input[patient_input < 50] = 50

    except Exception:
        if not (type(pred_horizon) == int or pred_horizon == None) or pred_horizon not in [1, 3, 7, None]:
            error_code = 101 # horizonte de predicao deve ser 1, 3, 7 ou None
        elif pd.isna(patient_input).any():
            error_code = 201 # dados observados nao podem conter NaN
        elif not all(isinstance(x[0], (int, float, np.int64)) for x in patient_input):
            error_code = 202 # dados observados devem conter apenas inteiros ou float
        elif len(patient_input) not in [*range(3, 11)]:
            error_code = 203 # tamanho do array dos dados observados entre 3 e 10 dias 

    return error_code, patient_input 
