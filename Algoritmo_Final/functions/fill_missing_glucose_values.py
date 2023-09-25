import pandas as pd
import numpy as np

def fill_missing_glucose_values(df):
    for i in range (len(df)-1):

        if pd.notnull(df['BG Reading (mg/dL)'][i]):
            df['Sensor Glucose (mg/dL)'][i] = df['BG Reading (mg/dL)'][i]
        
        elif (i != 0 and (pd.notnull(df['Sensor Glucose (mg/dL)'][i-1]) and pd.notnull(df['Sensor Glucose (mg/dL)'][i+1]) and np.isnan(df['BG Reading (mg/dL)'][i]))):
            df['Sensor Glucose (mg/dL)'][i] = round((df['Sensor Glucose (mg/dL)'][i-1]+df['Sensor Glucose (mg/dL)'][i+1])/2)

    #print("Numero de Nan values --> ",df['Sensor Glucose (mg/dL)'].isna().sum())
    df = df[df['Timestamp'].notna() & df['Sensor Glucose (mg/dL)'].notna()]
    df = df.reset_index(drop=True)

    return df