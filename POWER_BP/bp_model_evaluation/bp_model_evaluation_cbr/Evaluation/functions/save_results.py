import numpy as np
import pandas as pd
from pathlib import Path
import os

def save_results(results, method, pred_horizon, mape, mse, rmse, retrieved, adaptation, distancefcn):

    arr = np.array([method, pred_horizon, mape, mse, rmse, retrieved, adaptation, distancefcn])

    if len(results) == 0:
        if Path('results.csv').is_file() and os.stat('results.csv').st_size != 0:
            results = pd.read_csv('results.csv', header=0).values
            results = np.vstack([results, arr])
        else:
            results = arr

    else:
        results = np.vstack([results, arr])

    return results
