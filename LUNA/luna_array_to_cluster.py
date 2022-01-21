import numpy as np

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler


def scaling(array):
    """
    Scales an array using sklearn.
    """
    scale = StandardScaler()
    return scale.fit_transform(array)


def k_means(array, n=5, scaled=True):
    """
    Clusters an array using the K-means algorithm from sklearn.
    """
    # 1. scale array if required.
    if scaled:
        array = scaling(array)

    # 2. cluster array.
    model = cl.KMeans(n_clusters=n, random_state=42)
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    return cluster.reshape(-1, 1), cluster_values


def array_to_cluster(array_time_left, array_time_right, array_length_left, array_length_right):
    """
    Takes 4 gradient arrays as input and returns the clustered arrays for the left and right foot.
    """
    # 1. stack left and right foot.
    array_time = np.hstack((array_time_left, array_time_right))
    array_length = np.hstack((array_length_left, array_length_right))

    # 2. cluster time and length separately.
    cluster_time, cluster_time_values = k_means(array_time.reshape(-1, 1))
    cluster_length, cluster_length_values = k_means(array_length. reshape(-1, 1))

    # 3. merge clusters and split into left and right foot.
    cluster = cluster_time.reshape(array_time.shape) + cluster_length.reshape(array_length.shape)
    split = array_time_left.shape[1]
    cluster_left, cluster_right = cluster[:, :split], cluster[:, split:]

    return cluster_left, cluster_right
