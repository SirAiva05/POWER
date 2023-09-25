import pandas as pd
import os
import glob
pd.options.mode.chained_assignment = None  # default='warn'

from functions.fill_missing_glucose_values import fill_missing_glucose_values

def load_data(rootdir, test_index):

    all_files = glob.glob(rootdir + "/*.xlsx")
    all_files = sorted(all_files)
    #escolha do paciente para teste
    df_test = pd.read_excel(all_files[test_index], header=11)
    name = all_files[test_index] #guardar o nome do paciente de teste
    print("Dataset em analise -> ", name)
    #del all_files[test_index] #remover o nome do paciente de teste
    li = []
    for filename in all_files:
        df_int = pd.read_excel(filename, index_col=None, header=11)
        li.append(df_int)
    df = pd.concat(li, axis=0, ignore_index=True)
    cols = ['Index', 'Date', 'Time', 'Source', 'Excluded', 'Used in Calibration', 'ISIG Value', 'Sensor Event', 'Other', 'Raw-Type', 'Raw-Values','Carb Amount (grams)', 'Insulin Type', 'Insulin Units', 'Exercise Level', 'Sleep Start Time','Sleep Wake-up Time', 'Notes']
    #'Meal', 'Medication', 'Exercise', 'BG Reading (mg/dL)'
    df.drop(cols, axis=1, errors='ignore', inplace=True)
    df_test.drop(cols, axis=1, errors='ignore', inplace=True)

    df = fill_missing_glucose_values(df)
    df_test = fill_missing_glucose_values(df_test)
    
    
    return df, df_test
