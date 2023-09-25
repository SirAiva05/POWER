import numpy as np

# separar os dados de cada paciente em arrays de tamanho n para input (X), a cada qual estará associado o output real (y) 
def data_prep(patient_data, n, p, dayx):
    X, y = [], []
    for i in range(len(patient_data)):
        end_ix = i + n  # encontrar o final do padrão

        if not dayx: # output varios dias 0:P
            if end_ix > len(patient_data)-1:
                break
            if np.isnan(np.sum(patient_data[i:end_ix])) or np.isnan(patient_data[end_ix]):
                continue
            # input e output do padrão
            seq_x, seq_y = patient_data[i:end_ix], patient_data[end_ix]
            
        elif dayx: # output um unico dia P
            output_ix = end_ix + p - 1 # output
            if output_ix > len(patient_data)-1:
                break
            if np.isnan(np.sum(patient_data[i:end_ix])) or np.isnan(patient_data[output_ix]):
                continue
            # input e output do padrão
            seq_x, seq_y = patient_data[i:end_ix], patient_data[output_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)