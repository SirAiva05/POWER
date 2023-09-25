import math
import random

import matplotlib
import pandas as pd
import numpy as np


from pyampute.ampute import MultivariateAmputation

from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_auc_score
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, IsolationForest
from sklearn.linear_model import BayesianRidge
from sklearn.cluster import DBSCAN
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

import datetime
import matplotlib.pyplot as plt


def create_missing_values_simple(df, prob=0.3):
    mask = np.random.choice([True, False], size=df.shape, p=[prob, 1 - prob])
    x_nan = df.mask(mask, other=-1)
    x_nan = x_nan.to_numpy()
    return x_nan


def create_missing_values_multivariate(df, prob, patterns):
    # seed = 2022
    # np.random.RandomState(seed)

    ma = MultivariateAmputation(prop=prob, patterns=patterns)

    x = df.to_numpy()
    x_nan = ma.fit_transform(x)

    return x_nan


def create_outliers_multivariate(df, percentage, patterns, increment):
    # seed = 2022
    # np.random.RandomState(seed)

    ma = MultivariateAmputation(prop=percentage, patterns=patterns)

    x = df.to_numpy()
    x_nan = ma.fit_transform(x)
    mask = np.zeros(x_nan.shape)
    for iy, ix in np.ndindex(x_nan.shape):
        if np.isnan(x_nan[iy, ix]):
            mask[iy, ix] = 1
            choice = random.choice(["sum", "subtraction"])
            if choice == "sum":
                x[iy, ix] = x[iy, ix] + x[iy, ix] * increment
            else:
                x[iy, ix] = x[iy, ix] + x[iy, ix] * increment

    df_outlier = pd.DataFrame(x, columns=df.columns, index=df.index)

    return df_outlier, mask


def create_outliers_simple(df, percentage, increment=0.5):
    # np.random.seed(50)
    mask = np.random.choice([True, False], size=df.shape, p=[percentage, 1 - percentage])
    df_outlier = df.copy()
    df_outlier.reset_index(drop=True, inplace=True)
    column_id = 0
    for i in df_outlier.columns:
        for index, row in df_outlier.iterrows():
            print(index)
            if mask[index, column_id] == 1:
                choice = random.choice(["sum", "subtraction"])
                if choice == "sum":
                    df_outlier.iloc[index, column_id] = df_outlier.iloc[index, column_id] + df_outlier.iloc[
                        index, column_id] * increment
                elif choice == "subtraction":
                    df_outlier.iloc[index, column_id] = df_outlier.iloc[index, column_id] - df_outlier.iloc[
                        index, column_id] * increment
        column_id += 1
    df_outlier.index = df.index
    return df_outlier, mask


def simple_imputation_mean(df):
    x = df.to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    x = imp.fit_transform(x)
    df_nan = pd.DataFrame(x, columns=df.columns)

    return df_nan


def knn_impute(df):
    x = df.to_numpy()
    knn = KNNImputer(n_neighbors=20)
    knn.fit(x)
    x = knn.transform(x)
    df_nan = pd.DataFrame(x, columns=df.columns)

    return df_nan


def mice(df):
    x = df.to_numpy()
    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr, missing_values=np.nan, tol=1e-10, max_iter=10, verbose=0,
                           imputation_order='roman')
    x = imp.fit_transform(x)
    df_nan = pd.DataFrame(x, columns=df.columns)

    return df_nan


def miss_forest(df):
    x = df.to_numpy()
    regr = RandomForestRegressor(n_estimators=10, max_depth=20, bootstrap=True, max_samples=1, n_jobs=2)
    imp = IterativeImputer(estimator=regr, missing_values=np.nan, tol=1e-10, max_iter=100, verbose=0,
                           imputation_order='roman',
                           random_state=0)
    x = imp.fit_transform(x)
    df_nan = pd.DataFrame(x, columns=df.columns)
    return df_nan


def arima_prediction(df, p=5, d=0, q=1):
    df_filled = df.copy()
    df_filled = df_filled.interpolate()
    df_filled = df_filled.interpolate().fillna(method='bfill')
    df_filled = df_filled.interpolate().fillna(method='ffill')
    count = 0
    res = ""
    for col in df_filled:
        x = df_filled[col].to_numpy()
        model = ARIMA(x, order=(p, d, q))
        model_fit = model.fit()
        results = model_fit.fittedvalues
        if count == 0:
            res = results
            count += 1
        else:
            res = np.vstack((res, results))

    df_predicted = pd.DataFrame(res.T, columns=df.columns)
    df_predicted.index = df.index
    df_predicted.iloc[:p, :] = df_filled.iloc[:p, :]
    df_nan = df.copy()
    df_nan[df_nan.isnull()] = df_predicted

    return df_nan


def detect_outliers_iqr(df):
    res = np.full(df.shape, False)
    x = df.to_numpy()

    for c in range(x.shape[1]):
        Q1 = np.percentile(x[:, c], 25, method="midpoint")
        Q3 = np.percentile(x[:, c], 75, method="midpoint")
        IQR = Q3 - Q1

        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR

        for r in range(x.shape[0]):
            value = x[r, c]
            if value <= low_lim or value >= up_lim:
                res[r, c] = True

    res = res.sum(axis=1)
    res[res > 1] = 1

    return res


def dbscan_outliers(df):
    x = df.to_numpy()
    dbscan = DBSCAN(eps=5, metric="euclidean", min_samples=5, n_jobs=2)
    pred = dbscan.fit_predict(x)
    pred[pred == 1] = 0
    pred[pred == -1] = 1

    return pred


def isolation_forest(df):
    x = df.to_numpy()
    model = IsolationForest(n_estimators=256, max_samples=100, contamination='auto',
                            random_state=None)
    pred = model.fit_predict(x)
    pred[pred == 1] = 0
    pred[pred == -1] = 1

    return pred


# calculates the root mean squared error
def get_missing_values_results(real, predicted):
    mse = mean_squared_error(real.values, predicted.values)
    rmse = mse ** 0.5

    return rmse


# calculates the area under the curve
def get_outlier_results(real, predicted):
    real = real * 1
    real = real.sum(axis=1)
    real[real > 1] = 1

    auc = roc_auc_score(real.flatten(), predicted.flatten())

    return auc


# function to determine the rankings for the missing value imputation algorithms
def pipeline_training_miss(df, patterns):
    univariate = False
    if df.shape[1] == 1:
        univariate = True

    if univariate:
        x_nan = create_missing_values_simple(df, 0.3)
    else:
        x_nan = create_missing_values_multivariate(df, 0.3, patterns)

    df_nan = pd.DataFrame(x_nan, columns=df.columns, index=df.index)
    df_nan[df_nan < 0] = np.nan

    res_nan = simple_imputation_mean(df_nan)
    res_nan2 = knn_impute(df_nan)
    res_nan3 = miss_forest(df_nan)
    res_nan4 = arima_prediction(df_nan, 1, 0, 1)
    res_nan5 = mice(df_nan)

    rmse_mean = get_missing_values_results(df, res_nan)
    rmse_arima = get_missing_values_results(df, res_nan4)
    rmse_knn = get_missing_values_results(df, res_nan2)
    rmse_forest = get_missing_values_results(df, res_nan3)
    rmse_mice = get_missing_values_results(df, res_nan5)

    res = {"mean": rmse_mean, "arima": rmse_arima, "knn": rmse_knn, "missforest": rmse_forest, "mice": rmse_mice}
    return res


# function to determine the rankings for the outlier detection algorithms
def pipeline_training_out(df, patterns):
    univariate = False
    if df.shape[1] == 1:
        univariate = True
    if univariate:
        df_outlier, mask = create_outliers_simple(df, 0.1)
    else:
        df_outlier, mask = create_outliers_multivariate(df, 0.1, patterns, 0.5)

    res1 = detect_outliers_iqr(df_outlier)
    res2 = dbscan_outliers(df_outlier)
    res3 = isolation_forest(df_outlier)
    print(mask, res2)
    auc_iqr = get_outlier_results(mask, res1)
    auc_dbscan = get_outlier_results(mask, res2)
    auc_iso = get_outlier_results(mask, res3)

    res = {"iqr": auc_iqr, "dbscan": auc_dbscan, "isolation_forest": auc_iso}
    return res
