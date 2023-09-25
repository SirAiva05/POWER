import numpy as np
# pip install pydist2
# from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

def CBRretrieve(query, CB_input, number_of_retrieved_cases, distanceFunction, include_meal_information, n): #arrays

    target = query[-n:]
    case_meal = query[0]
    query_case = np.reshape(query[0:-n], (1, len(query[0:-n])))

    CB_cases = CB_input[:,0:-n]
    CB_cases_meal = CB_input[:, 0]
    CB_solutions = CB_input[:, -n:]

    if (include_meal_information):
        iii = np.where(CB_cases_meal == case_meal)[0]
        CB = CB_cases[iii, :]
        CB_sol = CB_solutions[iii]
    else:
        CB = CB_cases
        CB_sol = CB_solutions

    if (distanceFunction == 'correlation'):
        if (np.std(query_case) == 0):
            query_case[-n:] = query_case[-n:] + 0.000000001
        CB[np.std(np.transpose(CB))==0,-n:] = CB[np.std(np.transpose(CB))==0,-n:] + 0.000000001
        distance = pairwise_distances(query_case, CB, metric = distanceFunction)
        # hist(distance, 20)
    else:
        #print(query_case.shape)
        #print(CB.shape)
        distance = pairwise_distances(query_case, CB, metric = distanceFunction)
        # hist(distance, 20)

    sortedDistances = np.sort(np.transpose(distance), axis=0)
    idxSorted = np.argsort(np.transpose(distance), axis=0)

    sortedDistances = sortedDistances[0:number_of_retrieved_cases]
    idx = idxSorted[0:number_of_retrieved_cases,:]
    
    a = CB_sol[idx].squeeze()
    retrieved_cases = []
    if (include_meal_information):
        retrieved_cases = np.column_stack((np.ones((number_of_retrieved_cases, 1))*case_meal, np.reshape(CB[idx], (-n, CB.shape[1])), CB_sol[idx]))
    else:
        retrieved_cases = np.column_stack((np.reshape(CB[idx], (-n, CB.shape[1])), CB_sol[idx].squeeze()))
    distances = sortedDistances

    return retrieved_cases, distances