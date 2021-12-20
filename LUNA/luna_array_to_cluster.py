import numpy as np

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


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


def silhouette_score(array, labels):  # if needed could be called independent
    score = metrics.silhouette_score(array, labels)
    return score


def calinski_score(array, labels):  # if needed could be called independent
    score = metrics.calinski_harabasz_score(array, labels)
    return score


def davies_score(array, labels):  # if needed could be called independent
    score = metrics.davies_bouldin_score(array, labels)
    return score


def array_to_cluster(array_time_left, array_time_right, array_length_left, array_length_right):
    array_time = np.hstack((array_time_left, array_time_right))
    array_length = np.hstack((array_length_left, array_length_right))

    cluster_time, cluster_time_values = k_means(array_time.reshape(-1, 1))
    cluster_length, cluster_length_values = k_means(array_length. reshape(-1, 1))

    cluster = cluster_time.reshape(array_time.shape) + cluster_length.reshape(array_length.shape)
    split = array_time_left.shape[1]
    cluster_left, cluster_right = cluster[:, :split], cluster[:, split:]

    return cluster_left, cluster_right


def cluster_to_image(cluster):
    cluster_image = np.ones(cluster.shape)

    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            if not cluster[i, j]:
                cluster_image[i, j] = 0.0

    return np.flip(cluster_image, axis=0)


def print_scores_of_clusters(array, labels, panel_name, cluster_name, get_silhouette=True, get_calinski=True, get_davies=True):
    print(f"\nThese are the scores for panel {panel_name}, for cluster {cluster_name}")
    if get_silhouette:
        silhouette = silhouette_score(array, labels)
        print(f'----------------------------')
        print(f'silhouette score = {silhouette}')

    if get_calinski:
        calinski = calinski_score(array, labels)
        print(f'----------------------------')
        print(f'calinski score = {calinski}')

    if get_davies:
        davies = davies_score(array, labels)
        print(f'----------------------------')
        print(f'davies score = {davies}')

    print(f'----------------------------')
    return
