import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from PZT.load_pzt import StatePZT


def analyse_pzt(pzt_database, graphing=False):
    # for every run we will do a seperate analysis
    count = 0
    for run in pzt_database:
        count += 1
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

        ###########################################
        # # # Code of Niels # # #
        ###########################################

        if count < 2:
            continue
        freq_select = 100000
        all_features = ['max_amp', 'min_amp', 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time',
                        'travel_time']
        all_channels = ["Actionneur1", "Actionneur2", "Actionneur3", "Actionneur4", "Actionneur5", "Actionneur6",
                        "Actionneur7", "Actionneur8"]
        for channel_select in all_channels:
            fig, axs = plt.subplots(2, 4)  # y, x
            fig.suptitle(f'different features for emitter {channel_select} and with a frequency {freq_select}', fontsize=16)
            # initialize
            counter = 0
            for feature_select in all_features:
                state_to_plot = np.array([])
                for state_select in list(range(1, len(frequency_array_dict[freq_select]) + 1)):
                    feature_output = get_feature(frequency_array_dict, state_select, freq_select, channel_select,
                                                 feature_select)
                    if state_to_plot.shape == (0,):
                        state_to_plot = feature_output
                    else:
                        state_to_plot = np.vstack((state_to_plot, feature_output))
                x_counter = counter % 4
                y_counter = counter//4
                axs[y_counter, x_counter].plot(state_to_plot)
                axs[y_counter, x_counter].legend(['emitter', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8'])
                axs[y_counter, x_counter].set_title(feature_select)
                counter += 1
            plt.show()




        #
        # fig, axs = plt.subplots(2, 4)  # y, x
        #
        # axs[0, 0].plot(original_data, state)
        # axs[0, 0].set_title('Axis [0, 0]')
        #
        # #  feature 1
        # axs[0, 1].plot(features_plot[0], y, 'tab:orange')
        # axs[0, 1].set_title('Axis [0, 1]')
        # #  feature 2
        # axs[1, 0].plot(x, -y, 'tab:green')
        # axs[1, 0].set_title('Axis [1, 0]')
        # #  feature 3
        # axs[1, 1].plot(x, -y, 'tab:red')
        # axs[1, 1].set_title('Axis [1, 1]')



def get_feature(freq_dict, state, freq_select, channel_select, feature_select):
    """select a frequency and state, select an channel and a feature
        returns the feature as a np.array and also the current state"""
    state_select = state - 1
    features_dict_for_each_channel = freq_dict[freq_select][state_select][1]

    channel_df = features_dict_for_each_channel[channel_select]
    feature_output = channel_df[feature_select]
    return np.array(feature_output)
