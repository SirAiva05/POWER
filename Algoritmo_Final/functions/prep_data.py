import numpy as np

def prep_data(dataset, target, start_index, end_index, history_size, target_size):
    data=[]
    labels=[]
    
    start_index = start_index + history_size
    
    if end_index is None:
        end_index = len(dataset)-target_size
        
    for i in range(start_index,end_index):
        indices = range(i-history_size,i)
        data.append(np.reshape(dataset[indices],(history_size,1)))
        labels.append(target[i+target_size])
        
    return np.array(data),np.array(labels)


