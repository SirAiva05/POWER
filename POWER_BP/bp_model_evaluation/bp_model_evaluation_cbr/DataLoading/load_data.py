from operator import index
import os
import pandas as pd
from DataLoading.insert_missing import insert_missing


def load_data(csvpath):
    data = []
    splitbreak = []


    for filename in os.listdir(csvpath):
        df = pd.read_csv(csvpath + '/' + filename, header=None).astype('double')
        df = insert_missing(df)
        df = df.T
        variable_names = []
        for n in range(1, df.shape[1] + 1):
            variable_names.append('day'+str(n))
        df.columns = variable_names
        # length = len(df.index)
        length = df.shape[1]
        data.append(df)
        splitbreak.append(length)

    df_data = pd.concat(data, axis = 0, ignore_index=True)
    return df_data, splitbreak
