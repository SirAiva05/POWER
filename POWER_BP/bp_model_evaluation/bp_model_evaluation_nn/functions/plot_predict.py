import numpy as np
import matplotlib.pyplot as plt
import os

# plot para comparacao da predicao com os valores reais e save da imagem em png
def plot_predict(true, predicted, test_index, idx, method, n, m, dayx):
    
    if dayx:
        X = np.array(range(1, len(true)+1))
        plt.plot(X, true, X[n+m-1:], predicted)
    else:
        X = np.array(range(idx+1+n, idx+n+len(true)+1, 1))
        plt.plot(X, true, X, predicted)

    plt.title("Prediction: Patient " + str(test_index+1))
    plt.xlabel("Day")
    plt.ylabel("Systolic Blood Pressure (mmHg)")
    plt.legend(["True", "Predicted"])

    if dayx:
        dir = "plots/" + method + '/' + str(n) + '/' + str(m) + '/' + str(dayx)
    else:
        dir = "plots/" + method + '/' + str(n) + '/' + str(m) + '/' + str(dayx) + '/patient' + str(test_index + 1)

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    if dayx:
        plt.savefig(dir + "/patient" + str(test_index+1) + ".png")
    else:
        plt.savefig(dir + "/" + str(idx+1) + ".png")
    plt.cla()
