import os
import scipy.io
from matplotlib import pyplot as plt


location = "Files/L1-03/PZT/L103_2019_12_06_17_35_05/State_1_2019_12_06_17_35_05/250kHz_5cycles"

for actionneur in os.listdir(location):
    print(actionneur)
    y = os.listdir(location+'/'+actionneur)
    # print(y)
    # mat = scipy.io.loadmat(f"{location}/{actionneur}/{x}")
    # Matlab files loaded as dictionary files
    z = [scipy.io.loadmat(f"{location}/{actionneur}/{x}") for x in y]
    z = [x['Time_Response'] for x in z]
    #print(z[0].shape)

    # Matlab Graph
    if False:
        j = 0
        time_lst = z[j][:, 0]
        for i in range(1, 9):
            plt.plot(time_lst, z[j][:, i])
        plt.title(actionneur)
        plt.xlabel('Time')
        plt.show()

    # Weird graph
    if True:
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)
        fig.suptitle(actionneur)

        for i in range(1, 9):
            # len(z) = 10
            for j in range(len(z)):
                exec(f"ax{i}.plot(z[{j}][:,0],z[{j}][:,{i}])")
                exec(f"ax{i}.set_xlabel('Time')")
                if i != 1:
                    exec(f"ax{i}.set_title('Channel {i}')")
                else:
                    exec(f"ax{i}.set_title('Emission')")
        plt.show()



