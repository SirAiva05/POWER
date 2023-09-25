import pandas as pd
import numpy as np

def validate_glucose_data(patient_input, pred_horizon):
    
    patient_input = pd.DataFrame(patient_input).values
        
    try:
        error_code = 0
    
        patient_input[patient_input < 20] = 20
        patient_input[patient_input > 800] = 800
        
        if not all(isinstance(x[0], (int, float, np.int64)) for x in patient_input)\
            or pd.isna(patient_input).any() \
            or ((pred_horizon == 2 or pred_horizon == 4 or pred_horizon == None) and len(patient_input) < 120) \
            or (pred_horizon == 12 and len(patient_input) < 264)\
                or not (type(pred_horizon) == int or pred_horizon == None) or pred_horizon not in [None, 2, 4, 12]:
            
            raise Exception
            
    except Exception:
        
        if not (type(pred_horizon) == int or pred_horizon == None) or pred_horizon not in [None, 2, 4, 12]:
            error_code = 101
        elif pd.isna(patient_input).any(): 
            error_code = 201
        elif not all(isinstance(x[0], (int, float, np.int64)) for x in patient_input):
            error_code = 202
        elif pred_horizon in [None, 2] and len(patient_input) < 24:
            error_code = 203
        elif pred_horizon == 4 and len(patient_input) < 48:
            error_code = 204
        elif pred_horizon == 12 and len(patient_input) < 144:
            error_code = 205

    return error_code, patient_input



