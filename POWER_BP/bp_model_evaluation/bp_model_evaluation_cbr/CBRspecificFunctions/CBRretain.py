import numpy as np
def CBRretain(new_case, confirmed_solution, CB, memory_update_operation_mode):

    if (memory_update_operation_mode == 0):
       updated_CB = CB
    elif (memory_update_operation_mode == 1):
        updated_CB = np.vstack((CB, new_case))
    else:
        print('Something is wrong!')
    
    return updated_CB
