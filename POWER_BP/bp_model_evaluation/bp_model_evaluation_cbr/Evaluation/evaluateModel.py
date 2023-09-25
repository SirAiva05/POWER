import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
#from matplotlib.axis import Axis as axis
from sklearn.metrics import mean_absolute_error as mae

def timeGain(y, y_hat, p):
    
    if(y.ndim==1):
        y = np.transpose(y)
        y_hat = np.transpose(y_hat)
      
    #Y = np.array([y,]*(p+1))
    #Y = np.tile(y, (1, p+1))
    Y = np.transpose([y] * (p+1))
    Y = Y[0:-1-p+1,:] 
    Y_HAT = np.zeros((len(y)-p, p+1))
    for i in range(0,p):
        Y_HAT[:, i] = y_hat[i:-1-(p-i)+1]
        if (0):
            plt.plot(range(1,-1-p), 'r--')
            plt.show()
            plt.plot(range(i+1,-1-(p-i)), 'bo')
            plt.pause()
            plt.clf()
            
    delay = np.subtract(Y, Y_HAT)
    delay = np.power(delay, 2)
    delay = np.sum(delay, axis=0)/np.size(delay,1)
    delay = np.argmin(delay)
    TG = p - delay
    
    return TG


def ESODnorm(y, y_hat):

    x = y
    L = len(y)
    ESOD = []
    for i in range(2, L):
        ESOD.append((x[i] - 2 * x[i-1] + x[i-2]) ** 2)
    ESOD_y = np.sum(ESOD)

    x = y_hat
    L = len(y)
    ESOD = []
    for i in range(2, L):
        ESOD.append((x[i] - 2 * x[i-1] + x[i-2]) ** 2)
    ESOD_y_hat = np.sum(ESOD)

    ESODnormalized = ESOD_y_hat/ESOD_y

    return ESODnormalized


def clarke(y, yp):

    # Error checking
    if y is None or yp is None:
        print('clarke:Inputs', 'There are no inputs.')

    if len(yp) != len(y):
        print('clarke:Inputs', 'Vectors y and yp must be the same length.')

    if (np.amax(y) > 400) or (np.amax(yp) > 400) or (np.amin(y) < 0) or (np.amin(yp) < 0):
        print('clarke:Inputs',
              'Vectors y and yp are not in the physiological range of glucose (<400mg/dl).')

    # -------------------------- Print figure flag ---------------------------------
    PRINT_FIGURE = True
    # ------------------------- Determine data length ------------------------------
    n = len(y)
    # ------------------------- Plot Clarke's Error Grid ---------------------------
    h = plt.figure

    plt.plot(y, yp, 'ko', markersize=4,
             markerfacecolor='k', markeredgecolor='k')
    plt.xlabel('Reference Concentration [mg/dl]')
    plt.ylabel('Predicted Concentration [mg/dl]')
    plt.title('Clarke''s Error Grid Analysis')
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    # Theoretical 45º regression line
    plt.plot([0, 400], [0, 400], 'k:')
    plt.plot([0, 175/3], [70, 70], 'k-')
    # plot([175/3 320],[70 400],'k-')
    # replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([175/3, 400/1.2], [70, 400], 'k-')
    plt.plot([70, 70], [84, 400], 'k-')
    plt.plot([0, 70], [180, 180], 'k-')
    # Corrected upper B-C boundary
    plt.plot([70, 290], [180, 400], 'k-')
    # plot([70 70],[0 175/3],'k-')
    # replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    plt.plot([70, 70], [0, 56], 'k-')
    # plot([70 400],[175/3 320],'k-')
    plt.plot([70, 400], [56, 320], 'k-')
    plt.plot([180, 180], [0, 70], 'k-')
    plt.plot([180, 400], [70, 70], 'k-')
    plt.plot([240, 240], [70, 180], 'k-')
    plt.plot([240, 400], [180, 180], 'k-')
    # Lower B-C boundary slope OK
    plt.plot([130, 180], [0, 70], 'k-')
    plt.text(30, 20, 'A', fontsize=12)
    plt.text(30, 150, 'D', fontsize=12)
    plt.text(30, 380, 'E', fontsize=12)
    plt.text(150, 380, 'C', fontsize=12)
    plt.text(160, 20, 'C', fontsize=12)
    plt.text(380, 20, 'E', fontsize=12)
    plt.text(380, 120, 'D', fontsize=12)
    plt.text(380, 260, 'B', fontsize=12)
    plt.text(280, 380, 'B', fontsize=12)
    plt.show()

    ####################################### ERROS NOS AXIS DOS PLOT

    #axis.set(h, color = 'white')
    # sets the color to white
    # Specify window units
    #axis.set(h, 'units', 'inches')

    # Change figure and paper size (Fixed to 3x3 in)
    #axis.set(h, 'Position', [0.1, 0.1, 3, 3])
    #axis.set(h, 'PaperPosition', [0.1, 0.1, 3, 3])

    if PRINT_FIGURE:
        # Saves plot as a Enhanced MetaFile
        # print(h,'-dmeta','Clarke_EGA');
        # Saves plot as PNG at 300 dpi
        print(h, '-dpng', 'Clarke_EGA', '-r300')

    total = np.zeros((5, 1))  # Initializes output

    # ------------------------------- Statistics -----------------------------------
    for i in range(1, n):
        if (yp[i] <= 70 and y[i] <= 70) or (yp[i] <= 1.2*y[i] and yp[i] >= 0.8*y[i]):
            total[1] = total[1] + 1           # Zone A
        else:
            if ((y[i] >= 180) and (yp[i] <= 70)) or ((y[i] <= 70) and yp[i] >= 180):
                total[5] = total[5] + 1      # Zone E
            else:
                if ((y[i] >= 70 and y[i] <= 290) and (yp[i] >= y[i] + 110)) or ((y[i] >= 130 and y[i] <= 180) and (yp[i] <= (7/5)*y[i] - 182)):
                    total[3] = total[3] + 1    # Zone C
                else:
                    if ((y[i] >= 240) and ((yp[i] >= 70) and (yp[i] <= 180))) or (y[i] <= 175/3 and (yp[i] <= 180) and (yp[i] >= 70)) or ((y[i] >= 175/3 and y[i] <= 70) and (yp[i] >= (6/5)*y[i])):
                        total[4] = total[4] + 1  # Zone D
                    else:
                        total[2] = total[2] + 1  # Zone B

    percentage = (total/n)*100
    # -------------------------------------------------------------------------------
    # EOF

    return total, percentage


def evaluateModel(y, y_hat, splitbreak):

    # MAE = []
    # RMSE = []
    # MAPE = []
    MAE = 0
    RMSE = 0
    MAPE = 0
    TG = []
    ESOD = []
    J = []
    A = []
    B = []

    segment_begin = 0

    for i in range(0, len(splitbreak)):
        # segment_begin
        segment_end = segment_begin + splitbreak[i] #- 1

        yyy = y # [segment_begin:segment_end]
        yyy_hat = y_hat #[segment_begin:segment_end]

        #tg = timeGain(yyy, yyy_hat, 6)
        # esod = ESODnorm(yyy, yyy_hat)
        (yyy_hat > 200).choose(yyy_hat, 200)
        (yyy_hat < 50).choose(yyy_hat, 50)
        # total, percentage = clarke(yyy, yyy_hat)

        # MAE.append(mae(yyy, yyy_hat))
        # RMSE.append(np.sqrt(np.mean((yyy-yyy_hat)**2)))
        # MAPE.append(np.mean(abs((yyy-yyy_hat) / yyy)))
        #TG.append(tg)
        # ESOD.append(esod)
        #J.append(esod/(tg**2))
        #A.append(percentage[1])
        #B.append(percentage[2])

        MAE = mae(yyy, yyy_hat)
        RMSE = np.sqrt(np.mean((yyy-yyy_hat)**2))
        MAPE = np.mean(abs((yyy-yyy_hat) / yyy)) * 100

        segment_begin = splitbreak[i] #+ 1
    
    '''headers = ['MAE', 'RMSE', 'MAPE', 'TG', 'ESODnorm', 'J', 'A', 'B']
    m = [[MAE], [RMSE], [MAPE], [TG], [ESODnorm], [J], [A], [B]]
    evalTable = tabulate(m, headers, tablefmt="fancy_grid")'''


    # table = print(
    #                '___________________________\n',
    #               '|  MAE |', [round(x, 3) for x in MAE], '\n',
    #               '| RMSE |', [round(x, 3) for x in RMSE], '\n',
    #               '| MAPE |', [round(x, 3) for x in MAPE], '\n',
    #               '|  TG  |', [round(x, 3) for x in TG], '\n',
    #               '| ESOD |', [round(x, 3) for x in ESOD], '\n'
    #               ' |   J  |', [round(x, 3) for x in J], '\n',
    #               '|   A  |', [round(x, 3) for x in np.ravel(A)], '\n',
    #               '|   B  |', [round(x, 3) for x in np.ravel(B)], '\n',
    #               '___________________________\n')

    # return table
    return MAPE, RMSE, MAE
