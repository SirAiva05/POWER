import numpy as np

def trend_data(arr, trend_size):

    # arr_reduced = print(np.mean(np.pad(arr.astype(float), (0, trend_size - arr.size%int(trend_size)), 
    #                         mode='constant', 
    #                         constant_values=np.mean(arr)).reshape(-1, trend_size), axis=1))
        
    # if arr.size%int(trend_size)==0:
    #     arr_reduced = arr_reduced[:-1]
        
    arr_reduced = np.mean(arr.reshape(-1, trend_size), axis=1)
        
    return np.array(arr_reduced)




