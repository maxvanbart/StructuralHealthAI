from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


def init_clustering(database, delta=100, debug=False, debug_graph=False):
    # select the features for the clustering
    features = database.hits[database.hits["channel"] == 1]
    # features = features[:50000]
    # print(list(features))

    batches = batch_split(features, delta, debug=debug)

    # print some information about the batches
    if debug:
        print(len(batches))
        for batch in batches:
            print(batch.shape)

    # plot all datapoints which were selected in the features variable
    time, rms = features[["time"]], features["rms"]
    # plt.scatter(time, rms, s=4, c=features["channel"])

    # cluster all the batches found by the batch splitter
    for batch in tqdm(batches):
        if batch.shape[0] > 1:
            batch_cluster(batch, debug=debug, debug_graph=True)
        else:
            print('Batch contains a single object, skipping clustering...')

    n = 0
    if debug_graph:
        for batch in batches:
            n += len(batch)
            plt.scatter(batch[:, 0], batch[:, 6], s=4)

        print(n)
        plt.xlabel("Time")
        plt.ylabel("RMS voltage")
        plt.show()
    X = np.transpose(np.array([features["time"], features["rms"]]))
    # agglomerative(X, time, rms, features)


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
            if len(batch) > 30000 and delta > 10:
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
    clustering = model.fit(batch[:, [0, 6]])

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
    # print(batch)
    batch = np.array(batch)
    if debug:
        print(f"Amount of datapoints: {n_points}")
        print(f"Amount of clusters: {max(labels)}")
        print(f"That is {100*round((n_points-len(clusterdict))/n_points,3)}% less datapoints...")

    if debug_graph:
        plt.scatter(batch[:, 0], batch[:, 6], s=4, c=clustering.labels_)
        plt.xlabel("Time")
        plt.ylabel("RMS voltage")
        for i, label in enumerate(labels):
            plt.annotate(f"{int(batch[:, 11][i])}", (batch[:, 0][i], batch[:, 6][i]))
        plt.show()
    return batch
