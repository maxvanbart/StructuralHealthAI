from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import psutil


def init_clustering(database, delta=100, debug=False, debug_graph=False):
    # select the features for the clustering
    cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time', 'counts', 'channel', 'abs_time']
    features = database[cols]
    features = features[features["counts"] >= 2]

    # Extract the header and the channels as important variables
    channels = features['channel'].unique()
    header = list(features)
    if debug:
        print(channels)

    # Find the available memory and use it to determine the maximum cluster size
    # Larger maximum clusters will avoid clusters getting split up
    available_memory = psutil.virtual_memory()[0] / 1024 ** 3
    print(f"Detected {round(available_memory,1)}GB of system memory...")
    max_size = 20000
    if available_memory > 30:
        max_size = 30000
    elif available_memory < 10:
        max_size = 10000

    # Here we cluster all the datapoints per channel
    combined_batches = []
    for channel in channels:
        features_channel = features[features["channel"] == channel]
        batches = batch_split(features_channel, delta, debug=debug, max_size=max_size)

        # print some information about the batches if debug is enabled
        if debug:
            print(len(batches))
            for batch in batches:
                print(batch.shape)

        # Enabling this debug graph will show the batch division of the selected datapoints
        if debug_graph:
            n = 0
            for batch in batches:
                n += len(batch)
                plt.scatter(batch[:, 0], batch[:, 4], s=4)
            print(n)
            plt.title("Batch division of datapoints")
            plt.xlabel("Time [s]")
            plt.ylabel("RMS voltage [µV]")
            plt.show()

        # cluster all the batches found by the batch splitter
        for batch in tqdm(batches, desc=f"Channel {channel}"):
            if batch.shape[0] > 1:
                clustered_batch = batch_cluster(batch, debug=debug, debug_graph=debug_graph)
                combined_batches.append(batch_combine_points(clustered_batch, debug=debug))
            else:
                combined_batches.append(batch)

    # Combine all the clustered batches back together and return them
    combined_database = pd.DataFrame(np.vstack(combined_batches), columns=header)
    return combined_database


def batch_split(df, delta, dynamic_splitting=True, debug=False, max_size=20000):
    """This function takes a pandas dataframe and converts it into batches ready for clustering"""
    if type(df) == pd.core.frame.DataFrame:
        matrix = df.values.tolist()
    else:
        matrix = df

    # use this line to avoid overly small values for delta
    if delta < 1:
        delta = 1

    # here we put all the objects in the matrix in a batch which applies to the specified value of delta
    batches = []
    for row in matrix:
        # insert the first row if the batches list is empty
        if len(batches) == 0:
            batches.append([row])
        # after inserting the first row we use the delta to decide if we put it in the latest batch
        # or if we should start a new batch
        elif abs(row[0] - batches[-1][-1][0]) > delta:
            batches.append([row])
        else:
            batches[-1].append(row)

    # we look through the batches to see if there are any that will not fit in memory during clustering
    if dynamic_splitting:
        final_batches = []
        for batch in batches:
            # if we find a large batch we will recursively decrease the delta until we find a batch size which works
            # the cutoff for dynamic splitting should still be tweaked as 30000 might not be optimal
            if len(batch) > max_size and delta > 10:
                if debug:
                    print(f"Found batch of size {len(batch)}, splitting...")
                final_batches += batch_split(batch, delta-10, max_size=max_size)
            else:
                final_batches.append(batch)
    else:
        final_batches = batches
    return [np.array(batch) for batch in final_batches]


def batch_cluster(batch, debug=False, debug_graph=False):
    """This function uses the agglomerative clustering algorithm to cluster datapoints within a batch"""
    # time 0, RMS 6
    t0 = time.time()
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01, compute_full_tree=True)
    clustering = model.fit(batch[:, [0, 4]])

    t1 = time.time()
    if debug:
        print(f"Time elapsed while clustering: {round(t1 - t0, 3)} seconds")

    # generate final result
    labels = clustering.labels_
    n_points = len(batch)

    batch = list(batch)
    for i in range(len(labels)):
        batch[i] = list(batch[i])
        batch[i].append(labels[i])
    batch = np.array(batch)

    # General debug information
    if debug:
        print(f"Amount of datapoints: {n_points}")
        print(f"Amount of clusters: {max(labels)}")
        print(f"That is {100*round((n_points-max(labels))/n_points,3)}% less datapoints...")

    # Plot a graph which shows the datapoints with labels
    if debug_graph:
        for Annote in [True, False]:
            plt.scatter(batch[:, 0], batch[:, 4], s=4, c=clustering.labels_)
            plt.title("Clustered datapoints")
            plt.xlabel("Time [s]")
            plt.ylabel("RMS voltage [µV]")
            if Annote:
                for i, label in enumerate(labels):
                    plt.annotate(f"{int(batch[:, -1][i])}", (batch[:, 0][i], batch[:, 4][i]))
            plt.show()
    return batch


def batch_combine_points(batch, debug=False):
    """Function to combine the objects in a cluster to a single object"""
    batch = list(batch)
    cluster_dict = {}
    # First we sort the objects by their cluster number
    for obj in batch:
        if int(obj[-1]) in cluster_dict:
            cluster_dict[int(obj[-1])].append(obj)
        else:
            cluster_dict[int(obj[-1])] = [obj]

    # For every cluster in the cluster dictionary we replace it by a single datapoint
    matrix = []
    for cluster in cluster_dict:
        lst = []
        for obj in cluster_dict[cluster]:
            lst.append(obj)
        array = np.array(lst)
        if array.shape[0] == 1:
            # This is a 2d matrix, so we flatten it (removed cluster index)
            matrix.append(array.flatten()[:-1])
        # Combination of the clusters into new points
        else:
            # ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time','counts']
            # [avg, high, avg, high, same, from high amp, avg]
            final_array = list()
            # take the average time
            final_array.append(sum(array[:, 0]) / len(array[:, 0]))
            # take the maximum amplitude
            final_array.append(max(array[:, 1]))
            max_amp_index = list(array[:, 1]).index(max(array[:, 1]))
            # take the average of the duration
            final_array.append(sum(array[:, 2]) / len(array[:, 2]))
            # take the maximum energy
            final_array.append(max(array[:, 3]))
            # take the RMS of the first instance
            final_array.append(array[0, 4])
            # take the rise time of the maximum amplitude index
            final_array.append(array[:, 5][max_amp_index])
            # take the average of the counts
            final_array.append(sum(array[:, 6]) / len(array[:, 6]))
            # add the cluster number of the first index
            final_array.append(array[:, 7][0])
            # take the average absolute time
            final_array.append(sum(array[:, 8]) / len(array[:, 8]))
            final_array = np.array(final_array)

            matrix.append(final_array)

    # Finally, we turn the matrix into a numpy array (index column already removed in the creation of the matrix)
    matrix = np.array(matrix)
    return matrix
