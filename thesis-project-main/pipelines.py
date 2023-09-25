import logging

import numpy as np
import pandas as pd

from utils import *

from rank_funcs import *


def pre_processing_pipeline_training(dataset, cols, patient_id=None, path=""):
    error_code = 0
    if dataset is None:
        logging.error("Dataset is None. Dataset must be an array")
        error_code = 400
        return None, error_code
    if not isinstance(dataset, np.ndarray) and not isinstance(dataset, list):
        logging.error("Dataset is not of type array. Dataset must be an array")
        error_code = 401
        return None, error_code
    if not isinstance(cols, list) and len(cols) != len(dataset[0]) - 1:
        logging.warning("Columns must be an array")
        temp = list(range(len(dataset[0])))
        cols = [str(x) for x in temp]
    if patient_id is None or not isinstance(patient_id, int):
        logging.warning("patient_id must be of type int")
        patient_id = ""
    if not isinstance(path, str):
        logging.warning("path must be of type string")
        path = ""

    df = pd.DataFrame(dataset, columns=cols)
    df.set_index(cols[0], inplace=True)

    cols.pop(0)
    cols.sort()
    patient_id = str(patient_id)

    filename = patient_id
    patterns = []

    col_num = 0
    for col in cols:
        filename = filename + col
        patterns.append({"incomplete_vars": [col_num], "mechanism": "MAR"})
        col_num += 1
    dict_out = pipeline_training_out(df, patterns)
    dict_miss = pipeline_training_miss(df, patterns)
    rankings = {"outlier": dict_out, "missing": dict_miss}

    error_code = save_rankings(rankings, path + filename + ".json")

    return rankings, error_code


def pre_processing_pipeline(dataset, cols, patient_id=None, path=""):
    error_code = 0
    if dataset is None:
        logging.error("Dataset is None. Dataset must be an array")
        error_code = 400
        return dataset, error_code
    if not isinstance(dataset, np.ndarray) and not isinstance(dataset, list):
        logging.error("Dataset is not of type array. Dataset must be an array")
        error_code = 401
        return dataset, error_code
    if not isinstance(cols, list) and len(cols) != len(dataset[0]) - 1:
        logging.warning("Columns must be an array")
        temp = list(range(len(dataset[0])))
        cols = [str(x) for x in temp]
    if patient_id is None or not isinstance(patient_id, int):
        logging.warning("patient_id must be of type int")
        patient_id = ""
    if not isinstance(path, str):
        logging.warning("path must be of type string")
        path = ""

    df = pd.DataFrame(dataset, columns=cols)
    df.set_index(cols[0], inplace=True)

    cols.pop(0)
    cols.sort()
    cols = ''.join(cols)
    patient_id = str(patient_id)

    filename = patient_id + cols

    rankings, error_code = get_rankings(path+filename+".json")

    # try to get rankings based only on the cols
    if rankings is None:
        rankings, error_code = get_rankings(path+filename+".json")

    df_nan = df.copy()
    df_filled = df.copy()
    df_filled = df_filled.fillna(df.median())

    # multivariate without rankings
    if rankings is None and len(cols) > 1:
        first_rank_outlier = "dbscan"
        first_rank_miss = "missforest"
    # univariate without rankings
    elif rankings is None and len(cols) <= 1:
        first_rank_outlier = "iqr"
        first_rank_miss = "arima"
    else:
        outlier_ranks = rankings["outlier"]
        miss_ranks = rankings["missing"]

        first_rank_outlier = list(outlier_ranks.keys())[0]
        first_rank_miss = list(miss_ranks.keys())[0]

    # detect outliers
    if first_rank_outlier == "iqr":
        res = detect_outliers_iqr(df_filled)
    elif first_rank_outlier == "dbscan":
        res = dbscan_outliers(df_filled)
    else:
        res = isolation_forest(df_filled)

    # eliminate outliers
    df_nan['outlier'] = res
    df_nan.loc[df_nan['outlier'] == 1, :] = np.nan
    df_nan = df_nan.drop(["outlier"], axis=1)

    # impute missing values
    if first_rank_miss == "mean":
        res = simple_imputation_mean(df_nan)
    elif first_rank_miss == "arima":
        res = arima_prediction(df_nan)
    elif first_rank_miss == "mice":
        res = mice(df_nan)
    elif first_rank_miss == "missforest":
        res = miss_forest(df_nan)
    else:
        res = knn_impute(df_nan)

    return res.reset_index().values.tolist(), error_code
