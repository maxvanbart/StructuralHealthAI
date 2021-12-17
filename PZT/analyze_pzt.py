import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from PZT.load_pzt import StatePZT


def analyse_pzt(pzt_database):
    # for every run we will do a seperate analysis
    for run in pzt_database:
        # here we extract all the frequencies which are present in the data
        f_list = list()
        for state in pzt_database[run]:
            f_list += state.f_list
        f_list = list(set(f_list))

        # add an empty list to the dictionary for every frequency found
        frequency_array_dict = dict()
        for f in f_list:
            frequency_array_dict[f] = []

        # here we fill the frequency array dict with the results for the different frequencies
        for state in tqdm(pzt_database[run], desc='State'):
            z, state_number = state.analyse()
            for f in z:
                # tuple containing the state number (time value)
                # and the dictionary of actionneurs which contain dictionaries containing maximum amplitudes
                # per channel
                frequency_array_dict[f].append((state_number, z[f]))

        for f in frequency_array_dict:
            # Here we compile the data into an array to use for plotting
            # header = ['state', 'chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8']
            pre_array = []
            for s in frequency_array_dict[f]:
                lst = [s[0]] + list(s[1]['Actionneur1'].values())
                pre_array.append(lst)

            # convert to an array and sort by state (which represents time)
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
