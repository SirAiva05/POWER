from DataLoading.load_data import load_data
from CBRspecificFunctions.CBRreuse import CBRreuse
from CBRspecificFunctions.CBRretain import CBRretain
from CBRspecificFunctions.CBRrevise import CBRrevise
from CBRspecificFunctions.CBRretrieve import CBRretrieve
from Evaluation.evaluateModel import evaluateModel
from Evaluation.functions.save_results import save_results
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
import pandas as pd


# p = prediction horizon
# number_of_retrieved_cases = numero de casos semelhantes para serem comparados
# type_of_case_adaptation = tipo de funcao de adaptacao desses casos utilizada na obtencao de uma predicao
# distance_metric = funcao de distancia utilizada para a pesquisa de similaridade entre casos 
def CBR_predict_bp(p, number_of_retrieved_cases, type_of_case_adaptation, distance_metric):

    data_path = 'dataset'

    # testing_data_path = 'exampleDataset/test'

    data, splitbreak = load_data(data_path)
    n_patients = data.shape[0]

    patient_idx = [*range(0, np.size(data, 0), 1)]

    patients = data.values
    results = []

    # MODEL options

    PARAM_number_of_retrieved_cases = number_of_retrieved_cases

    # 0 - average, 1 - weighted average
    PARAM_type_of_case_adaptation = type_of_case_adaptation

    # 'euclidean', 'sqeuclidean', 'seuclidean', 'mahalanobis',
    # 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'correlation',
    # 'hamming', 'jaccard'
    PARAM_distance_metric = distance_metric

    # 0 - ignore meal input, 1 - include meal input
    PARAM_include_meal_information = 0

    # 0 - do not update CB, 1 - include all new cases
    PARAM_memory_update_operation_mode = 1
                
    data, splitbreak = load_data(data_path)
    # rows = patients | columns = days

    # Data loading / TRAIN

    # splitbreak is the vector with the lenght of each subject's data
    data, splitbreak = load_data(data_path)

    variable_names = list(data.columns.values)

    mape_score = []
    mse_score = []
    rmse_score = []
    mae_score = []

    for test_index in patient_idx:

        train_index = patient_idx.copy()
        train_index.remove(test_index)
        train_patients, test_patient = patients[train_index,
                                                :], patients[test_index, :]

        # training_data_meal = data.meal
        training_data_X = train_patients[:, 0:-p]
        training_data_Y = train_patients[:, -p:]

        # testing_data_meal = data.meal.to_numpy()
        testing_data_X = np.reshape(test_patient[0:-p], (1, len(test_patient[0:-p]))) 
        testing_data_Y = np.reshape(test_patient[-p:], (1, p)) 

        start_new_subject = np.cumsum(splitbreak)
        start_new_subject = np.array([[0], [start_new_subject[0]]])
        #start_new_subject = start_new_subject + 1

        new_cases = np.column_stack((testing_data_X, testing_data_Y))
        initial_CB = np.column_stack((training_data_X, training_data_Y))
        Y = testing_data_Y
        Y_hat = np.zeros(np.shape(Y))

        # del(data)
        # del(testing_data_X, testing_data_Y)
        # del(training_data_X, training_data_Y)

        # PREDICT case-by-case
        for i in range(0, len(new_cases)):

            # if (any(start_new_subject == i)):    # we start from the initial CB for each subject
            CB = initial_CB

            new_case = new_cases[i, :]

            retrieved_cases, distances = CBRretrieve(
                new_case, CB, PARAM_number_of_retrieved_cases, PARAM_distance_metric, PARAM_include_meal_information, p)
            proposed_solution = CBRreuse(
                new_case, retrieved_cases, distances, PARAM_type_of_case_adaptation, p)
            confirmed_solution = CBRrevise(proposed_solution)
            CB = CBRretain(new_case, confirmed_solution, CB,
                        PARAM_memory_update_operation_mode)

            Y_hat[i] = confirmed_solution

        # %% EVALUATE results
        # results_mod = evaluateModel(Y, Y_hat, splitbreak)
        # print(results_mod)
        MAPE, RMSE, MAE = evaluateModel(Y, Y_hat, splitbreak)

        mape_score.append(MAPE)
        rmse_score.append(RMSE)
        mae_score.append(MAE)

    result_mape = np.mean(np.absolute(mape_score))
    result_rmse = np.mean(np.absolute(rmse_score))
    result_mae = np.mean(np.absolute(mae_score))

    results = save_results(results, 'CBR', p, result_mape,
                        result_rmse, result_mae, PARAM_number_of_retrieved_cases, PARAM_type_of_case_adaptation, PARAM_distance_metric)

    header = ['method', 'P', 'MAPE', 'MAE', 'RMSE', 'NRetrieved', 'adaptation', 'distancefcn']

    if results.ndim == 1:
        pd.DataFrame(results).transpose().to_csv(
            "results.csv", index=False, header=header)
    else:
        pd.DataFrame(results).to_csv(
            "results.csv", index=False, header=header)


for p in [3, 5, 7, 9]:  # number of values to predict
    for number_of_retrieved_cases in [4, 6, 8]:
        for type_of_case_adaptation in [0, 1]:
            for distance_metric in ['euclidean', 'sqeuclidean', 'seuclidean', 'mahalanobis', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'correlation', 'hamming', 'jaccard']:
                CBR_predict_bp(p, number_of_retrieved_cases, type_of_case_adaptation, distance_metric)