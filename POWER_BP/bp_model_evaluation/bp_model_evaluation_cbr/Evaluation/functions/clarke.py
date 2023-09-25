import matplotlib.pyplot as plt
from matplotlib.axis import Axis as axis
import numpy as np

def clarke(y, yp):
    
    # Error checking
    if y is None or yp is None:
        print('clarke:Inputs','There are no inputs.')
    
    if len(yp) != len(y):
        print('clarke:Inputs','Vectors y and yp must be the same length.')

    if (max(y) > 400) or (max(yp) > 400) or (min(y) < 0) or (min(yp) < 0):
        print('clarke:Inputs','Vectors y and yp are not in the physiological range of glucose (<400mg/dl).')
    
    #-------------------------- Print figure flag ---------------------------------
    PRINT_FIGURE = True
    #------------------------- Determine data length ------------------------------
    n = len(y)
    #------------------------- Plot Clarke's Error Grid ---------------------------
    h = plt.figure
    
    plt.plot(y,yp,'ko',markersize=4,markerfacecolor ='k', markeredgecolor ='k')
    plt.xlabel('Reference Concentration [mg/dl]')
    plt.ylabel ('Predicted Concentration [mg/dl]')
    plt.title('Clarke''s Error Grid Analysis')
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.plot([0, 400],[0, 400],'k:')                  # Theoretical 45º regression line
    plt.plot([0, 175/3],[70, 70],'k-')
    #plot([175/3 320],[70 400],'k-')
    plt.plot([175/3, 400/1.2],[70, 400],'k-')         # replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70],[84, 400],'k-')
    plt.plot([0, 70],[180, 180],'k-')
    plt.plot([70, 290],[180, 400],'k-')               # Corrected upper B-C boundary
    #plot([70 70],[0 175/3],'k-')
    plt.plot([70, 70],[0, 56],'k-')                   # replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    #plot([70 400],[175/3 320],'k-')
    plt.plot([70, 400],[56, 320],'k-')
    plt.plot([180, 180],[0, 70],'k-')
    plt.plot([180, 400],[70, 70],'k-')
    plt.plot([240, 240],[70, 180],'k-')
    plt.plot([240, 400],[180, 180],'k-')
    plt.plot([130, 180],[0, 70],'k-')                 # Lower B-C boundary slope OK
    plt.text(30,20,'A',fontsize=12);
    plt.text(30,150,'D',fontsize=12);
    plt.text(30,380,'E',fontsize=12);
    plt.text(150,380,'C',fontsize=12);
    plt.text(160,20,'C',fontsize=12);
    plt.text(380,20,'E',fontsize=12);
    plt.text(380,120,'D',fontsize=12);
    plt.text(380,260,'B',fontsize=12);
    plt.text(280,380,'B',fontsize=12);
    plt.show()
    
    axis.set(h, 'color', 'white');  
                 # sets the color to white 
    #Specify window units
    axis.set(h, 'units', 'inches')
    
    #Change figure and paper size (Fixed to 3x3 in)
    axis.set(h, 'Position', [0.1, 0.1, 3, 3])
    axis.set(h, 'PaperPosition', [0.1, 0.1, 3, 3])
    
    if PRINT_FIGURE:
        #Saves plot as a Enhanced MetaFile
        #print(h,'-dmeta','Clarke_EGA');           
        #Saves plot as PNG at 300 dpi
        print(h, '-dpng', 'Clarke_EGA', '-r300'); 
        
    total = np.zeros((5,1))                      #Initializes output

    # ------------------------------- Statistics -----------------------------------
    for i in range(1, n):
        if (yp[i] <= 70 and y[i] <= 70) or (yp[i] <= 1.2*y[i] and yp[i] >= 0.8*y[i]):
            total[1] = total[1] + 1           # Zone A
        else:
            if ( (y[i] >= 180) and (yp[i] <= 70) ) or ( (y[i] <= 70) and yp[i] >= 180 ):
                total[5] = total[5] + 1      # Zone E
            else:
                if ((y[i] >= 70 and y[i] <= 290) and (yp[i] >= y[i] + 110) ) or ((y[i] >= 130 and y[i] <= 180) and (yp[i] <= (7/5)*y[i] - 182)):
                    total[3] = total[3] + 1    # Zone C
                else:
                    if ((y[i] >= 240) and ((yp[i] >= 70) and (yp[i] <= 180))) or (y[i] <= 175/3 and (yp[i] <= 180) and (yp[i] >= 70)) or ((y[i] >= 175/3 and y[i] <= 70) and (yp[i] >= (6/5)*y[i])):
                        total[4] = total[4] + 1 # Zone D
                    else:
                        total[2] = total[2] + 1 # Zone B

    percentage = (total/n)*100
    #-------------------------------------------------------------------------------
    #EOF
    
    return total, percentage
