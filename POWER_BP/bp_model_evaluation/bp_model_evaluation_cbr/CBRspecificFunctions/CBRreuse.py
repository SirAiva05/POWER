import numpy as np
from sklearn.linear_model import LinearRegression


def getWeightsFromDistancesVectorized(distances):

    idx_0 = np.where(distances == 0)[0]
    idx_else = np.where(distances != 0)[0]

    weights = np.zeros((len(distances),1))
    weights[idx_0] = 1
    weights[idx_else] = 1/(distances[idx_else])
    weights = weights/np.sum(weights)
    weights = weights.ravel()

    return weights


def CBRreuse(new_case, retrieved_cases, distances, type_of_case_adaptation, n):

    proposed_solution_n = np.zeros((1,n))
    solution_n = retrieved_cases[:,-n:]
    for i in range(n):
        query_case = new_case[0:-n]
        CB = retrieved_cases[:, 0:-n]
        solution = solution_n[:,i]
        if (type_of_case_adaptation != 3):
            # Option 1: remove mean
            solution = solution - np.transpose(np.mean(CB, axis=1)) + np.mean(query_case)
            # Option 2: remove delta last element
            # solution = solution - CB_cases[idx[i,:], -1] + cases[i,-1]
        # distance = sortedDistances

        if (type_of_case_adaptation == 1):  # weighted
            w = getWeightsFromDistancesVectorized(distances)
            proposed_solution = np.sum(w*solution)
        elif(type_of_case_adaptation == 0):  # not weighted
            proposed_solution = np.mean(solution)
        # elif(type_of_case_adaptation == 3): # linear regression
        #    LRproblem = CB[idx, :]
        #    LRsol = CB_sol[idx]
        #    LRquery = cases[i,:]
        #    proposed_solution = LinearRegression().fit(LRproblem, LRsol, query_case)
        proposed_solution_n[0,i] = proposed_solution
    return proposed_solution_n
