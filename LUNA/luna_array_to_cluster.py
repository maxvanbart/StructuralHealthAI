import numpy as np

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler


def scaling(array):
    scale = StandardScaler()
    return scale.fit_transform(array)


def k_means(array, n=5, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.KMeans(n_clusters=n, random_state=42)
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def array_to_cluster(array_time_left, array_time_right, array_length_left, array_length_right):
    array_time = np.hstack((array_time_left, array_time_right))
    array_length = np.hstack((array_length_left, array_length_right))

    cluster_time, cluster_time_values = k_means(array_time.reshape(-1, 1))
    cluster_length, cluster_length_values = k_means(array_length. reshape(-1, 1))

    cluster = cluster_time.reshape(array_time.shape) + cluster_length.reshape(array_length.shape)
    split = array_time_left.shape[1]
    cluster_left, cluster_right = cluster[:, :split], cluster[:, split:]

    return cluster_left, cluster_right
