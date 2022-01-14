import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import sklearn.cluster as cls

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
    col_list = ["state", "frequency", "actionneur"] + selected_features

    # only select frequency
    freq_filtered = database.loc[database['frequency'] == selected_frequency]

    # merge into output list
    selected_data = freq_filtered[col_list]

    # create a list of dataframes that contain the features per actionneur
    cluster_list = []
    for act in selected_data["actionneur"].drop_duplicates():
        act_lst = []
        act_data = selected_data.loc[selected_data["actionneur"] == act]
        for state in act_data["state"].drop_duplicates():
            state_data = act_data.loc[act_data["state"] == state]
            act_lst.append(state_data)
        cluster_list.append(act_data)

    # create the array for the clustering
    names = []  # ["kmeans n=4", "kmeans n=7", ...]
    all_cluster_labels = []
    n_algorithms = 0
    act_lst = []
    for ndx, act in enumerate(cluster_list):
        act_lst = []
        for state in act["state"].drop_duplicates():
            state_data = act.loc[act["state"] == state]
            act_lst.append(state_data.to_numpy().flatten())
        kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.1), random_state=42)
        kmean_cluster.fit(act_lst)
        kmean_labels1 = kmean_cluster.labels_
        all_cluster_labels.append(kmean_labels1)
        name = f'kmeans n={int(len(act_lst)*0.1)}'
        names.append(name)

        kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.2), random_state=42)
        kmean_cluster.fit(act_lst)
        kmean_labels2 = kmean_cluster.labels_
        all_cluster_labels.append(kmean_labels2)
        name = f'kmeans n={int(len(act_lst)*0.2)}'
        names.append(name)

        kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.3), random_state=42)
        kmean_cluster.fit(act_lst)
        kmean_labels3 = kmean_cluster.labels_
        all_cluster_labels.append(kmean_labels3)
        name = f'kmeans n={int(len(act_lst)*0.3)}'
        names.append(name)

        aff_prop_cluster = cls.AffinityPropagation()
        aff_prop_cluster.fit(act_lst)
        aff_prop_labels = aff_prop_cluster.labels_
        all_cluster_labels.append(aff_prop_labels)
        name = "aff_prop"
        names.append(name)

        optics_cluster = cls.OPTICS()
        optics_cluster.fit(act_lst)
        optics_labels = optics_cluster.labels_
        all_cluster_labels.append(optics_labels)
        name = "OPTICS"
        names.append(name)

        if n_algorithms == 0:
            n_algorithms = len(names)

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

    total_sum_list = []
    for i in range(n_algorithms):
        list_to_add = change_array[i::n_algorithms]
        sum_list = np.sum(list_to_add, axis=0)
        total_sum_list.append(sum_list)

    change_df = pd.DataFrame(data=np.array(total_sum_list).T, columns=names[0:n_algorithms], index=range(1, len(act_lst)+1))
    output_sum = np.sum(total_sum_list, axis=0)

    if barplot:
        ax = change_df.plot.bar(rot=1, stacked=True)
        plt.title("State vs amount of cluster hits")
        plt.suptitle(f'For the following features: {selected_features}')
        plt.xlabel("State.no")
        plt.ylabel("Amount of clusters")
        plt.plot(output_sum, ":", c="tab:brown")
        plt.show()


# ---------------------------------
# output pzt----
# --------------------------------
# type 1: state(s) -> [3, 4, 5, 9, 10] -- total 5
# type 2: state(s) -> 1, 2, 7, 8 -- total of 4
# type 3: state(s) -> 11, 12, 13, 14 -- total of 4
# -----------------------------------------------
