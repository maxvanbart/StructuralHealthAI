import numpy as np
from sklearn import metrics


# This is the scoring code made by the LUNA team
def silhouette_score(array, labels):  # if needed could be called independent
    score = metrics.silhouette_score(array, labels)
    return score


def calinski_score(array, labels):  # if needed could be called independent
    score = metrics.calinski_harabasz_score(array, labels)
    return score


def davies_score(array, labels):  # if needed could be called independent
    score = metrics.davies_bouldin_score(array, labels)
    return score


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