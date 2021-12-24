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
        #  # * # * #   Code of Niels   # * # * #  #
        ###########################################
        if count < 2:
            continue
        # should be possible to make this into a neater function
        all_frequency = [50000, 100000, 125000, 150000, 200000, 250000]
        all_frequency = [200000]
        all_features = ['max_amp', 'min_amp', 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time',
                        'travel_time']
        all_channels = ["Actionneur1", "Actionneur2", "Actionneur3", "Actionneur4", "Actionneur5", "Actionneur6",
                        "Actionneur7", "Actionneur8"]
        all_channels = ["Actionneur1"]
        gradient = False
        plotting = True

        for freq_select in all_frequency:  # loop over all the different frequencies
            for channel_select in all_channels:  # loop over all of the channels
                fig, axs = plt.subplots(2, 4)  # y, x
                fig.suptitle(f'different features for emitter {channel_select} and with a frequency {freq_select}',
                             fontsize=16)
                counter = 0  # counter to know where to plot the plot
                for feature_select in all_features:  # loop over features, max of 8 features possible
                    state_to_plot = np.array([])
                    for state_select in list(range(1, len(frequency_array_dict[freq_select]) + 1)):
                        # loop over all the states, start at state 1 till end
                        feature_output = get_feature(frequency_array_dict, state_select, freq_select, channel_select,
                                                     feature_select)
                        # function to get all of the features for selected parameters
                        if state_to_plot.shape == (0,):  # if empty initialize
                            state_to_plot = feature_output
                        else:  # else go stacking for different states
                            state_to_plot = np.vstack((state_to_plot, feature_output))
                    x_counter = counter % 4  # placement
                    y_counter = counter//4  # placement
                    if gradient is True:
                        # gradient gives 2 outputs for a 2d array not sure which one to pick
                        # they both give different results but same kind of interesting points can be observed,
                        # namely state 10 and between 20 and 25
                        axs[y_counter, x_counter].plot(np.gradient(state_to_plot)[1])
                        # so which one do we use?
                    else:
                        axs[y_counter, x_counter].plot(state_to_plot)
                    axs[y_counter, x_counter].legend(['emitter', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7',
                                                      'chan8'])
                    axs[y_counter, x_counter].set_title(feature_select)
                    counter += 1  # update counter for next subplot
                if plotting is True:
                    plt.show()


def get_feature(freq_dict, state, freq_select, channel_select, feature_select):
    """select a frequency and state, select an channel and a feature
        returns the feature as a np.array and also the current state"""
    state_select = state - 1
    features_dict_for_each_channel = freq_dict[freq_select][state_select][1]  # enter dictionary with freq and state

    channel_df = features_dict_for_each_channel[channel_select]  # get dataFrame with channel
    feature_output = channel_df[feature_select]  # get features output with selected feature
    return np.array(feature_output)  # convert to numpy array
