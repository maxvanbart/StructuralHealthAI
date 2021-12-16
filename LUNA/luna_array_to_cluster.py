import numpy as np

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler


def scaling(array):
    scale = StandardScaler()
    return scale.fit_transform(array)


def k_means(array, n=5):
    # scale array
    array_scaled = scaling(array)

    # clustering
    model = cl.KMeans(n_clusters=n, random_state=42)
    cluster = model.fit_predict(array_scaled)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def mean_shift(array):
    # scale array
    array_scaled = scaling(array)

    # clustering
    model = cl.MeanShift()
    cluster = model.fit_predict(array_scaled)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def aff_prop(array):
    # scale array
    array_scaled = scaling(array)

    # clustering
    model = cl.AffinityPropagation()
    cluster = model.fit_predict(array_scaled)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def agglo(array, n=5):
    # scale array
    array_scaled = scaling(array)

    # clustering
    model = cl.AgglomerativeClustering(n_clusters=n)
    cluster = model.fit_predict(array_scaled)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values

