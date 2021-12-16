import numpy as np

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler
from sklear.metrics import silhouette_score


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


def mean_shift(array, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.MeanShift()
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def aff_prop(array, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.AffinityPropagation()
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def agglo(array, n=5, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.AgglomerativeClustering(n_clusters=n)
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


# def get_scores():
#     if get_silhouette:
#         silhouette = silhouette_score((cluster, labels))
#     return
#
#
# def silhouette_score(cluster, labels):
#
#
#     score = silhouette_score(cluster, labels=labels, random_state=42)
#
#     return score