from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


def init_clustering(database, delta=100, debug=False, debug_graph=False):
    # select the features for the clustering
    cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time', 'counts']
    features = database.hits[cols]
    features = features[-500000:]
    # debug_graph = True

    # ['time', 'channel', 'param_id', 'amplitude', 'duration', 'energy', 'rms', 'threshold', 'rise_time', 'counts',
    # 'cascade_hits']
    header = list(features)

    batches = batch_split(features, delta, debug=debug)

    # print some information about the batches
    if debug:
        print(len(batches))
        for batch in batches:
            print(batch.shape)

    # Enabeling this debug graph will show the batch division of the selected datapoints
    if debug_graph:
        n = 0
        for batch in batches:
            n += len(batch)
            plt.scatter(batch[:, 0], batch[:, 4], s=4)
        print(n)
        plt.xlabel("Time")
        plt.ylabel("RMS voltage")
        plt.show()

    # cluster all the batches found by the batch splitter
    combined_batches = []
    for batch in tqdm(batches):
        if batch.shape[0] > 1:
            clustered_batch = batch_cluster(batch, debug=debug)
            combined_batches.append(batch_combine_points(clustered_batch, debug=debug))
        else:
            combined_batches.append(batch)

    combined_database = pd.DataFrame(np.vstack(combined_batches), columns=header)

    return combined_database


def batch_split(df, delta, dynamic_splitting=True, debug=False):
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
        if abs(row[0] - batches[-1][-1][0]) > delta:
            batches.append([row])
        else:
            batches[-1].append(row)

    # we look through the batches to see if there are any that will not fit in memory during clustering
    if dynamic_splitting:
        final_batches = []
        for batch in batches:
            # if we find a large batch we will recursively decrease the delta until we find a batch size which works
            # the cutoff for dynamic splitting should still be tweaked as 30000 might not be optimal
            if len(batch) > 25000 and delta > 10:
                if debug:
                    print(f"Found batch of size {len(batch)}, splitting...")
                final_batches += batch_split(batch, delta-10)
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
        print(f"That is {100*round((n_points-len(labels))/n_points,3)}% less datapoints...")

    # Plot a graph which shows the datapoints with labels
    if debug_graph:
        plt.scatter(batch[:, 0], batch[:, 4], s=4, c=clustering.labels_)
        plt.xlabel("Time")
        plt.ylabel("RMS voltage")
        for i, label in enumerate(labels):
            plt.annotate(f"{int(batch[:, 11][i])}", (batch[:, 0][i], batch[:, 6][i]))
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
            # This is a 2d matrix so we flatten it (removed cluster index)
            matrix.append(array.flatten()[:-1])
        # Combination of the clusters into new points
        else:
            # ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time','counts']
            # [avg, high, add, add, same, from high amp, add]
            final_array = []
            final_array.append(sum(array[:, 0]) / len(array[:, 0]))
            final_array.append(max(array[:,1]))
            max_amp_index = list(array[:,1]).index(max(array[:,1]))
            final_array.append(sum(array[:, 2]))
            final_array.append(sum(array[:, 3]))
            final_array.append(array[0, 4])
            final_array.append(array[:, 5][max_amp_index])
            final_array.append(sum(array[:, 6]))
            final_array = np.array(final_array)
            if debug:
                print(f"Mean for cluster {cluster}")
                print(array.shape)
                print(array)
            matrix.append(final_array)

    # Finally we turn the matrix into a numpy array (index column already removed in the creation of the matrix)
    matrix = np.array(matrix)
    return matrix
