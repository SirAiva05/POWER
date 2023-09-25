import numpy as np
import pandas as pd
from pathlib import Path
import os

# guardar resultados num ficheiro csv
def save_results(results, method, n, pred_horizon, pae, mse, rmse, mae, activationfcn, activationfcn1, optimizerfcn, lossfcn, dayx):


    arr = np.array([method, n, pred_horizon, pae, mse, rmse,
                   mae, activationfcn, activationfcn1, optimizerfcn, lossfcn, dayx])

    if len(results) == 0:
        if Path('results.csv').is_file() and os.stat('results.csv').st_size != 0:
            results = pd.read_csv('results.csv', header=0).values
            results = np.vstack([results, arr])
        else:
            results = arr

    else:
        results = np.vstack([results, arr])

    return results
