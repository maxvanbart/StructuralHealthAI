import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from PZT.load_pzt import StatePZT


def analyse_pzt(pzt_database, panel_name, graphing=False):

    # for every run we will do a seperate analysis
    count = 0
    for run in pzt_database:
        count += 1
        # /!\ MIGHT BE BETTER TO BASE THIS ON THE AMOUNT OF STATES IN A RUN /!\
        if count < 2:
            # This prevents a division by zero error
            continue
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
        # should be possible to make this into a neater function
        # all_frequency = [50000, 100000, 125000, 150000, 200000, 250000]
        # This list should in the final product only contrain the 'usefull frequencies'
        all_frequency = [50000, 100000, 125000, 150000, 200000, 250000]
        # Only usefull features should be contained in this list for the final product
        # all_features = ['max_amp', 'min_amp', 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time',
        #                 'travel_time', 'energy']
        all_features = ['relative_amp', 'duration', 'rise_time', 'travel_time', 'energy']
        all_channels = ["Actionneur1", "Actionneur2", "Actionneur3", "Actionneur4", "Actionneur5", "Actionneur6",
                        "Actionneur7", "Actionneur8"]
        # all_channels = ["Actionneur1"]
        gradient = False

        hits = {}
        for freq_select in all_frequency:  # loop over all the different frequencies

            for channel_select in all_channels:  # loop over all of the channels
                if channel_select not in hits:
                    hits[channel_select] = []
                outliers_channels = None
                header_channels = None

                if graphing:
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

                    # Some information about the data
                    averages = np.average(state_to_plot, axis=0)
                    standarddivs = np.std(state_to_plot, axis=0)

                    l1 = averages
                    l2 = averages + standarddivs
                    l3 = averages - standarddivs

                    state_filtered = np.copy(state_to_plot)
                    state_filtered = abs(state_filtered - averages)
                    state_filtered = state_filtered - standarddivs
                    outliers = state_filtered > 0
                    outliers = np.sum(outliers, axis=1)

                    # Pass the outlier data a level downward
                    if outliers_channels is None:
                        outliers_channels = outliers
                        header_channels = [feature_select]
                    else:
                        outliers_channels = np.vstack((outliers_channels, outliers))
                        header_channels.append(feature_select)

                    if graphing:
                        # Placement code
                        x_counter = counter % 4
                        y_counter = counter // 4

                        color_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                        color_lst += ['tab:brown', 'tab:pink', 'tab:gray']

                        axs[y_counter, x_counter].plot(state_to_plot)

                        line_width = state_to_plot.shape[0]

                        axs[y_counter, x_counter].hlines(l1, 0, line_width, colors=color_lst)
                        axs[y_counter, x_counter].hlines(l2, 0, line_width, linestyles='dashed', colors=color_lst)
                        axs[y_counter, x_counter].hlines(l3, 0, line_width, linestyles='dashed', colors=color_lst)

                        # axs[y_counter, x_counter].legend(['emitter', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7',
                        #                                   'chan8'])
                        axs[y_counter, x_counter].set_title(feature_select)
                    counter += 1  # update counter for next subplot

                # Here we prepare the generated data matrix for the next level
                if outliers_channels is not None:
                    outliers_channels = np.transpose(outliers_channels)
                    hits[channel_select].append(outliers_channels)

                if graphing:
                    plt.show()

        # Here we combine the dictionary of margin violations to pull conclusions about the timestamps
        # where changes in the panel properties occur
        hits_processed = {}
        for y in hits:
            hit = sum(hits[y])
            # hit = np.sum(hit, axis=1)
            # print(hit)
            hits_processed[y] = hit
            hits_df = pd.DataFrame(data=hit, columns=all_features)

            ax = hits_df.plot.bar(rot=1, stacked=True)
            plt.title(f'Margin violations for different measurements of {y} on panel {panel_name}.')
            plt.show()

            # for y in
            # plt.bar(range(hit.shape[0]), hit)
            # plt.title(f'Margin violations for different measurements of {y}.')
            # plt.show()

        print(hits_processed)


def get_feature(freq_dict, state, freq_select, channel_select, feature_select):
    """select a frequency and state, select a channel and a feature
        returns the feature as a np.array and also the current state"""
    state_select = state - 1
    features_dict_for_each_channel = freq_dict[freq_select][state_select][1]  # enter dictionary with freq and state

    channel_df = features_dict_for_each_channel[channel_select]  # get dataFrame with channel
    feature_output = channel_df[feature_select]  # get features output with selected feature
    return np.array(feature_output)  # convert to numpy array
