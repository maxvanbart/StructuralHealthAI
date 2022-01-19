import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import sklearn.cluster as cls


# this function only works if multiple states are present in the files. Set the count value correctly
def analyse_pzt(pzt_database, time_check=False):
    # for every run we will do a separate analysis
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
        # if time_check:
        #     plt.plot(time_list)
        #     plt.title("check if it is a strait line if not, time sync is wrong")
        #     plt.show()


def make_clusters(database, panel_name, results_dir, barplot):
    """
    input: database, panel name
    working: clusters all of the data on the selection below. If one emitter is not usefull remove from the list.
             if all_clusters_graph is True, shows a graph of all clusters, so it is visible what states belong together
             if barplot is True, shows the bar plot with interesting data and states
    returns: all of the interesting points of the clusters and the name of each cluster used.
    """
    all_clusters_graph = False  # plot one graph with cluster labels for emitter 1
    selected_frequencies = [250000]  # selected frequency
    selected_features = ['relative_amp', 'duration', "avg_freq"]

    # column list for selection in database
    col_list = ["state", "frequency", "actionneur"] + selected_features

    for frequency in selected_frequencies:
        # only select frequency
        freq_filtered = database.loc[database['frequency'] == frequency]

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

            # kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.1), random_state=42)
            # kmean_cluster.fit(act_lst)
            # kmean_labels1 = kmean_cluster.labels_
            # all_cluster_labels.append(kmean_labels1)
            # name = f'kmeans n={int(len(act_lst)*0.1)}'
            # names.append(name)

            kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.2), random_state=42)
            kmean_cluster.fit(act_lst)
            kmean_labels2 = kmean_cluster.labels_
            all_cluster_labels.append(kmean_labels2)
            name = f'kmeans n={int(len(act_lst)*0.2)}'
            names.append(name)

            # kmean_cluster = cls.KMeans(n_clusters=int(len(act_lst)*0.25), random_state=42)
            # kmean_cluster.fit(act_lst)
            # kmean_labels3 = kmean_cluster.labels_
            # all_cluster_labels.append(kmean_labels3)
            # name = f'kmeans n={int(len(act_lst)*0.25)}'
            # names.append(name)

            aff_prop_cluster = cls.AffinityPropagation(random_state=None)
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

        selected_emitter_plot = 1
        if all_clusters_graph:
            for i in range(len(names)):
                print(i)
                if (selected_emitter_plot-1) * n_algorithms <= i < selected_emitter_plot * n_algorithms:
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

        list_to_file = [[], [], []]
        # 90%, 50%, else
        selected_groups = [0.8, 0.5]
        for numb, item in enumerate(output_sum):
            if item > selected_groups[0]*max(output_sum):
                list_to_file[0].append(numb+1)
                continue
            elif item > selected_groups[1] * max(output_sum):
                list_to_file[1].append(numb + 1)
                continue
            list_to_file[2].append(numb+1)

        ax = change_df.plot.bar(rot=1, stacked=True, figsize=(9, 6))
        plt.title("Cumulative number of feature changes detected by an emitter between states \n "
                  f"Features selected: {selected_features} \n " 
                  f"Frequency selected: {frequency/1000} kHz, Panel: {panel_name}")
        plt.suptitle(f'')
        plt.xlabel("State number")
        plt.ylabel("Number of feature changes detected")
        plt.plot(output_sum, ":", c="tab:brown")
        plt.hlines(selected_groups[0]*max(output_sum), 0, len(act_lst), linestyles=":", color='tab:blue', label=f"{selected_groups[0]*100}%")
        plt.hlines(selected_groups[1]*max(output_sum), 0, len(act_lst), linestyles=":", color='tab:pink', label=f"{selected_groups[1]*100}%")
        plt.legend()

        plt.savefig(f'{results_dir}/PZT_statechanges_{panel_name}.png')

        if barplot:
            plt.show()

        string_to_file = "\n"
        string_to_file += "---------------------------------\n"
        string_to_file += f"Importance of changes detected in panel {panel_name} for frequency {frequency} at states: \n"
        string_to_file += f"(Used {selected_groups[0]*100}%, {selected_groups[1]*100}% as thresholds)\n"
        string_to_file += "---------------------------------\n"
        string_to_file += f"High interest: state(s) -> \t {list_to_file[0]} -- total of {len(list_to_file[0])} state(s)\n"
        string_to_file += f"Medium interest: state(s) -> {list_to_file[1]} -- total of {len(list_to_file[1])} state(s)\n"
        string_to_file += f"Low interest: state(s) -> \t {list_to_file[2]} -- total of {len(list_to_file[2])} state(s)\n"
        string_to_file += "---------------------------------\n"

        with open(results_dir+f"/PZT_clustering-output_{panel_name}.txt", "w+") as f:
            f.write(string_to_file)
