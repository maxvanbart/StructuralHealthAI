import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import sklearn.cluster as cls

from PZT.load_pzt import StatePZT


# this function only works if multiple states are present in the files. Set the count value correctly
def analyse_pzt(pzt_database, graphing=False, time_check=False):
    # for every run we will do a seperate analysis
    count = 0
    for run in sorted(pzt_database):
        count += 1
        # /!\ MIGHT BE BETTER TO BASE THIS ON THE AMOUNT OF STATES IN A RUN /!\
        if count < len(pzt_database):
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
        time_list = []
        pzt_database_run = sorted(pzt_database[run])
        for state in tqdm(pzt_database_run, desc='State'):
            z, state_number = state.analyse()
            for f in z:  # freq in dict
                # tuple containing the state number (time value)
                # and the dictionary of actionneurs which contain dictionaries containing maximum amplitudes
                # per channel

                frequency_array_dict[f].append((state_number, z[f]))

            # make the start_time list
            time_list.append(state.start_time)
        time_list = np.array(time_list)
        if time_check:
            plt.plot(time_list)
            plt.title("check if it is a strait line if not, time sync is wrong")
            plt.show()

        ###########################################
        #  # * # * #   Code of Niels   # * # * #  #
        ###########################################
        # should be possible to make this into a neater function, like input
        # all_frequency = [50000, 100000, 125000, 150000, 200000, 250000]
        # This list should in the final product only contain the 'useful frequencies'
        all_frequency = [250000]

        # Only useful features should be contained in this list for the final product
        # all_features = ['max_amp', 'min_amp', 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time',
        #                 'travel_time', 'energy']
        all_features = ['relative_amp', 'duration', 'rise_time', 'travel_time', 'energy', "avg_freq"]
        all_channels = ["Actionneur1", "Actionneur2", "Actionneur3", "Actionneur4", "Actionneur5", "Actionneur6",
                        "Actionneur7", "Actionneur8"]
        # all_channels = ["Actionneur1"]

        hits = {}
        for freq_select in all_frequency:  # loop over all the different frequencies
            for channel_select in all_channels:  # loop over all of the channels
                if channel_select not in hits:
                    hits[channel_select] = []
                outliers_channels = None

                if graphing:
                    fig, axs = plt.subplots(2, 4)  # y, x
                    fig.suptitle(f'different features for emitter {channel_select} and with a frequency {freq_select}',
                                 fontsize=16)
                counter = 0  # counter to know where to plot the plot
                for feature_select in all_features:  # loop over features, max of 8 features possible
                    state_to_plot = np.array([])

                    state_select_list = list(range(1, len(frequency_array_dict[freq_select]) + 1))
                    for state_select in state_select_list:

                        # loop over all the states, start at state 1 till end
                        feature_output = get_feature(frequency_array_dict, state_select, freq_select, channel_select,
                                                     feature_select)
                        # function to get all of the features for selected parameters
                        if state_to_plot.shape == (0,):  # if empty initialize
                            state_to_plot = feature_output
                        else:  # else go stacking for different states
                            state_to_plot = np.vstack((state_to_plot, feature_output))

                    counter += 1  # update counter for next subplot

                # Here we prepare the generated data matrix for the next level
                if outliers_channels is not None:
                    outliers_channels = np.transpose(outliers_channels)
                    hits[channel_select].append(outliers_channels)

                if graphing:
                    plt.show()

        # # Here we combine the dictionary of margin violations to pull conclusions about the timestamps
        # # where changes in the panel properties occur
        # hits_processed = {}
        # for y in hits:
        #     hit = sum(hits[y])
        #     # hit = np.sum(hit, axis=1)
        #     # print(hit)
        #     hits_processed[y] = hit
        #     hits_df = pd.DataFrame(data=hit, columns=all_features)
        #
        #     ax = hits_df.plot.bar(rot=1, stacked=True)
        #     plt.title(f'Margin violations for different measurements of {y} on panel {panel_name}.')
        #     plt.show()
        #
        #     # for y in
        #     # plt.bar(range(hit.shape[0]), hit)
        #     # plt.title(f'Margin violations for different measurements of {y}.')
        #     # plt.show()

        # return hits_processed


def get_feature(freq_dict, state, freq_select, channel_select, feature_select):
    """select a frequency and state, select a channel and a feature
        returns the feature as a np.array"""
    state_select = state - 1
    features_dict_for_each_channel = freq_dict[freq_select][state_select][1]  # enter dictionary with freq and state

    channel_df = features_dict_for_each_channel[channel_select]  # get dataFrame with channel
    feature_output = channel_df[feature_select]  # get features output with selected feature
    return np.array(feature_output)  # convert to numpy array


def make_clusters(database, all_clusters_graph=False, barplot=True):
    """
    input: database
    working: clusters all of the data on the selection below. If one emitter is not usefull remove from the list.
             if all_clusters_graph is True, shows a graph of all clusters, so it is visible what states belong together
             if barplot is True, shows the bar plot with interesting data and states
    returns: all of the interesting points of the clusters and the name of each cluster used.
    """
    selected_frequency = 250000
    selected_features = ['relative_amp', 'duration', "avg_freq"]

    # column list for selection in database
    col_list = ["state", "frequency"] + selected_features

    # only select frequency
    freq_filtered = database.loc[database['frequency'] == selected_frequency]

    # merge into output list
    selected_data = freq_filtered[col_list]


    pass
    # create the array for the clustering
    cluster_list_data = np.array(cluster_list_data)
    names = []  # ["kmeans n=4", "kmeans n=7", ...]

    all_cluster_labels = []
    # do the clustering itself
    kmean_cluster = cls.KMeans(n_clusters=int(len(state_select_list)*0.1), random_state=42)
    kmean_cluster.fit(cluster_list_data)
    kmean_labels1 = kmean_cluster.labels_
    all_cluster_labels.append(kmean_labels1)
    name = f'kmeans n={int(len(state_select_list)*0.1)}'
    names.append(name)

    kmean_cluster = cls.KMeans(n_clusters=int(len(state_select_list)*0.2), random_state=42)
    kmean_cluster.fit(cluster_list_data)
    kmean_labels2 = kmean_cluster.labels_
    all_cluster_labels.append(kmean_labels2)
    name = f'kmeans n={int(len(state_select_list)*0.2)}'
    names.append(name)

    kmean_cluster = cls.KMeans(n_clusters=int(len(state_select_list)*0.3), random_state=42)
    kmean_cluster.fit(cluster_list_data)
    kmean_labels3 = kmean_cluster.labels_
    all_cluster_labels.append(kmean_labels3)
    name = f'kmeans n={int(len(state_select_list)*0.3)}'
    names.append(name)

    aff_prop_cluster = cls.AffinityPropagation()
    aff_prop_cluster.fit(cluster_list_data)
    aff_prop_labels = aff_prop_cluster.labels_
    all_cluster_labels.append(aff_prop_labels)
    name = "aff_prop"
    names.append(name)

    optics_cluster = cls.OPTICS()
    optics_cluster.fit(cluster_list_data)
    optics_labels = optics_cluster.labels_
    all_cluster_labels.append(optics_labels)
    name = "OPTICS"
    names.append(name)

    if all_clusters_graph:
        for i in range(len(names)):
            plt.plot(all_cluster_labels[i], label=names[i])
        plt.xlabel("State no.")
        plt.ylabel("label no.")
        plt.title("All groups clusters, if change of label no. \n that means a change of data, so interesting point")
        plt.legend()
        plt.show()

    from itertools import tee, islice, chain

    def previous_and_next(some_iterable):
        prevs, items, nexts = tee(some_iterable, 3)
        prevs = chain([None], prevs)
        nexts = chain(islice(nexts, 1, None), [None])
        return zip(prevs, items, nexts)

    # get the interesting points of the clusters
    changelst = []
    for cluster_list in all_cluster_labels:
        changes = []

        for prev, item, nxt in previous_and_next(cluster_list):
            change = 0
            if prev is None:
                if nxt != item:
                    change = 1
            elif nxt is None:
                if prev != item:
                    change = 1
            else:
                if nxt != item or prev != item:
                    change = 1
            changes.append(change)

        changelst.append(changes)

    change_array = np.array(changelst)
    change_array_sum = np.sum(change_array, axis=0)

    change_df = pd.DataFrame(data=change_array.T, columns=names, index=range(1, len(state_select_list)+1))
    if barplot:
        ax = change_df.plot.bar(rot=1, stacked=True)
        plt.title("State vs amount of cluster hits")
        plt.xlabel("State.no")
        plt.ylabel("Amount of clusters")
        plt.plot(change_array_sum, c="tab:brown")
        plt.show()
    return change_array, names


# ---------------------------------
# output pzt----
# --------------------------------
# type 1: state(s) -> 3, 4, 5, 9, 10 -- total 5
# type 2: state(s) -> 1, 2, 7, 8 -- total of 4
# type 3: state(s) -> 11, 12, 13, 14 -- total of 4
# -----------------------------------------------
