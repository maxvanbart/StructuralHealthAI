from luna_data_to_array import raw_to_array
from luna_data_to_array import gradient_arrays

import numpy as np
import sklearn.cluster as cl


def k_means(panel):
    # get vector
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.KMeans(n_clusters=3, random_state=0)
    cluster = model.fit_predict(array_right_time)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def mean_shift(panel):
    # get vector
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.MeanShift(min_bin_freq=5, cluster_all=True)
    cluster = model.fit_predict(array_right_time)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def affinity_propagation(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.AffinityPropagation()
    cluster = model.fit_predict(array_right_time)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def agglomerative_clustering(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.AgglomerativeClustering(n_clusters=3)
    cluster = model.fit_predict(array_right_time)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values