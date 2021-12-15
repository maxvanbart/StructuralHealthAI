from LUNA.luna_data_to_array import raw_to_array
from LUNA.luna_data_to_array import gradient_arrays
import sklearn.cluster as cl


def k_means(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.KMeans(n_clusters=3, random_state=0)
    clusters = model.fit_predict(array_right_time)

    # output
    return clusters.reshape(-1, 1)


def mean_shift(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.MeanShift()
    cluster = model.fit_predict(array_right_time)

    # output
    return cluster.reshape(-1, 1)


def aff_prop(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model = cl.AffinityPropagation()
    clusters = model.fit_predict(array_right_time)

    # output
    return clusters.reshape(-1, 1)


def Agglo(panel):
    # get array
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)
    array_right_time, array_right_length = gradient_arrays(array_right)

    # clustering
    model - cl.AgglomerativeClustering()
    clusters = model.fit_predict(array_right_time)

    # output
    return clusters.reshape(-1, 1)

