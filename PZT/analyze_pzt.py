import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from PZT.load_pzt import StatePZT


def analyse_pzt(pzt_database):
    header = ['amplitude']

    # this code extracts all the unique frequencies from the state objects

    for run in pzt_database:
        f_list = list()
        for state in pzt_database[run]:
            f_list += state.f_list
        f_list = list(set(f_list))
        # print(f_list)

        frequency_array_dict = dict()
        for f in f_list:
            frequency_array_dict[f] = []

        for state in tqdm(pzt_database[run], desc='State'):
            z, state_number = state.analyse()
            for f in z:
                frequency_array_dict[f].append((state_number, z[f]))

        for f in frequency_array_dict:
            pre_array = []
            for s in frequency_array_dict[f]:
                # print(s)
                lst = [s[0]] + list(s[1]['Actionneur1'].values())
                pre_array.append(lst)

            X = np.array(pre_array)
            X = X[np.argsort(X[:, 0])]

            # Do some graphing
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)
            fig.suptitle(f"Max amp over states with f = {f}")

            # i indicates the i'th channel of the matrix
            for i in range(1, 9):
                exec(f"ax{i}.plot(X[:,0],X[:, {i}])")
                exec(f"ax{i}.set_xlabel('Time')")
                if i != 1:
                    exec(f"ax{i}.set_title('Channel {i}')")
                else:
                    exec(f"ax{i}.set_title('Emission')")
            plt.show()
