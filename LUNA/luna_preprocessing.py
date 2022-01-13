import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


def cluster_array(timestamps, plot=False):

    def plot_cluster():
        values = np.ones(len(timestamps))

        plt.scatter(timestamps, values, c=timestamps_clustered)
        plt.show()

    clustering = AgglomerativeClustering(n_clusters=None, linkage='single',
                                         distance_threshold=3000).fit(timestamps.reshape(-1, 1))
    timestamps_clustered = clustering.labels_

    if plot:
        plot_cluster()

    return timestamps_clustered


def split_array(timestamps, timestamps_clustered, array, length=18, plot=False):

    def plot_databases():
        for n in range(len(array_split)):
            first_database_luna = array_split[n]

            first_timestamps_luna = first_database_luna[:, 0]
            first_values_luna = np.ones(len(first_timestamps_luna))

            plt.scatter(first_timestamps_luna, first_values_luna)
            plt.show()

    timestamps_split = []
    array_split = []

    array[:, 0] = array[:, 0] - array[0, 0]

    count = 0
    previous_split = 0

    for i in range(len(timestamps_clustered) - 1):
        count += 1

        if timestamps_clustered[i + 1] != timestamps_clustered[i] and count >= length:
            timestamps_split.append(timestamps[previous_split: i + 1])
            array_split.append(array[previous_split: i + 1, :])
            count = 0
            previous_split = i + 1

    if len(array) - previous_split > length:
        array_split.append(array[previous_split: -1, :])
        timestamps_split.append(timestamps[previous_split: -1])

    if plot:
        plot_databases()

    return array_split


def filter_array(array, margin_start=20, margin_small=0.5, margin_medium=0.5, margin_big=0.1, length=18):

    def remove_outliers_start():
        cut_start = 0

        for i in range(len(timestamps)):
            if intervals_small + margin_start > intervals_np[i] > intervals_small - margin_start:
                cut_start = i
                break

        return timestamps[cut_start:], array[cut_start:]

    def remove_outliers_middle():
        intervals = [timestamps_filtered[i + 1] - timestamps_filtered[i] for i in range(len(timestamps_filtered) - 1)]
        intervals.insert(0, 0)

        index_to_be_removed = None

        for i in range(len(intervals)):

            if i % 2 == 0 and i != 0 and i % length != 0:
                if intervals[i] < margin_medium * intervals_medium \
                        or intervals[i] > (1 + intervals_medium) * intervals_medium:
                    index_to_be_removed = i
                    break

            elif i % 2 == 1:
                if intervals[i] < margin_small * intervals_small \
                        or intervals[i] > (1 + intervals_small) * intervals_small:
                    index_to_be_removed = i
                    break

            elif i != 0 and i % length == 0:
                if intervals[i] < margin_big * intervals_big or intervals[i] > (1 + margin_big) * intervals_big:
                    index_to_be_removed = i
                    break

        if index_to_be_removed is None:
            return timestamps_filtered, array_filtered, True
        else:
            return np.delete(timestamps_filtered, index_to_be_removed), \
                   np.delete(array_filtered, index_to_be_removed, 0), False

    # Getting timestamps from arrays.
    timestamps = array[:, 0]

    # Getting the intervals from timestamps.
    intervals_np = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    intervals_pd = pd.DataFrame(intervals_np, dtype=float)

    intervals_counts = [i[0] for i in intervals_pd.value_counts().index.tolist()]
    intervals_medium = np.max(intervals_counts[:2])
    intervals_small = np.min(intervals_counts[:2])
    intervals_big = np.mean([i for i in intervals_counts if i > 2 * intervals_medium])

    # Removing all outliers from database.
    timestamps_filtered, array_filtered = remove_outliers_start()
    timestamps_filtered, array_filtered, completed = remove_outliers_middle()

    while not completed:
        timestamps_filtered, array_filtered, completed = remove_outliers_middle()

    return array_filtered


def preprocess_array(array, margin_start=20, margin_small=0.5, margin_medium=0.5, margin_big=0.1, length=18, plot=False):

    timestamps = array[:, 0] - array[0, 0]
    timestamps_clustered = cluster_array(timestamps, plot)

    array_split_all = split_array(timestamps, timestamps_clustered, array, length, plot=False)
    array_split_completed = filter_array(array_split_all[0], margin_start, margin_small, margin_medium, margin_big, length)

    for m in range(1, len(array_split_all)):
        array_split = filter_array(array_split_all[m])
        array_split_completed = np.vstack((array_split_completed, array_split))

    if plot:
        timestamps = array_split_completed[:, 0] - array_split_completed[0, 0]
        values = np.ones(len(timestamps))

        plt.scatter(timestamps, values)
        plt.show()

    return array_split_completed
