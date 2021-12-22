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

        break
        ###########################################
        # # # Code of Niels # # #
        ###########################################

        features_to_plot = [1, 2, 3, 4, 5, 6, 7, 8]
        # [[2d], [2d], ... feature 7]
        for freq in frequency_array_dict:  # select freq -> get state and features dict for each channel
            state = frequency_array_dict[freq][0][0]
            features_dict_for_each_channel = frequency_array_dict[freq][0][1]
            for channel in features_dict_for_each_channel:  # select a channel -> get df
                print(channel)
                df = features_dict_for_each_channel[channel]
                for numb, feature in enumerate(df):     # iterate over features
                    features_to_plot[numb] = np.array(df[feature])
                for numb, feature in enumerate(features_to_plot):
                    plt.plot(feature, label=f"feature {numb}")
                plt.legend()
                plt.show()
            break  # to get features from one source channel
        break  # just to get one freq







        fig, axs = plt.subplots(2, 4)  # y, x

        axs[0, 0].plot(original_data, state)
        axs[0, 0].set_title('Axis [0, 0]')

        #  feature 1
        axs[0, 1].plot(features_plot[0], y, 'tab:orange')
        axs[0, 1].set_title('Axis [0, 1]')
        #  feature 2
        axs[1, 0].plot(x, -y, 'tab:green')
        axs[1, 0].set_title('Axis [1, 0]')
        #  feature 3
        axs[1, 1].plot(x, -y, 'tab:red')
        axs[1, 1].set_title('Axis [1, 1]')

        break  # just first map for shape
