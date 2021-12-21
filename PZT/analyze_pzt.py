import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from PZT.load_pzt import StatePZT


def analyse_pzt(pzt_database, graphing=False):
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
            # this header can be used with the generated array to get create a pandas dataframe
            # header = ['state', 'chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8']
            # pre_array = []
            for s in frequency_array_dict[f]:
                lst = [s[0]] + list(s[1]['Actionneur3'].values())
            # header = ['state', 'chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8']
            pre_array = []
            for s in frequency_array_dict[f]:
                lst = [s[0]] + list(s[1]['Actionneur1'].values())
                pre_array.append(lst)




            # convert to an array and sort by state (which represents time) -> max amplitude
            X = np.array(pre_array)
            X = X[np.argsort(X[:, 0])]

            if graphing is True:
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

        # get matlab_array for initial state

        #####################
        # Niels begins here #
        #####################

        z = state.get_matlab_array()
        print(z)

        for i in z:  # iterate over freq
            # convert to numpy array
            array_i = np.array(i.matlab_array)
            print(array_i.shape)

            avg_of_channels, time = get_avg_for_8channels(array_i)
            print(avg_of_channels.shape)
            plt.plot(avg_of_channels)
            plt.show()

        #         print(measure.shape)
        #         print(measure[0])
        #         if measure[0, -1] not in test_dict:
        #             test_dict[measure[0, -1]] = 1
        # print("last item of list =", test_dict)
        break


def get_avg_for_8channels(array):
    # input = array for
    avg_of_channels = []
    for channel_numb in range(0, 8):  # do it amount of times there are channels
        channel = []
        for numb, measure in enumerate(array):  # iterate over measurements
            time = measure[:, 0]
            channels = measure[:, 1:]  # get rid of time
            channel.append(channels[:, channel_numb])  # select channel 1
        channel = np.average(channel, axis=0)
        avg_of_channels.append(channel)
    avg_of_channels = np.array(avg_of_channels)
    return avg_of_channels.T, time
