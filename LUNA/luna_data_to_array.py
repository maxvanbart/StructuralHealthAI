import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import os

from sklearn.cluster import AgglomerativeClustering


def file_to_array(panel, path):
    """
    Opens file in default LUNA data format and converts this into left and right foot numpy arrays.
    """
    def read_sensor_file():
        sensor_file = '/LUNA_sensor.txt'
        sensor_data = {}

        with open(os.path.dirname(__file__) + sensor_file) as file:
            data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype=str)

            for line in data:
                sensor_data[line[0]] = [float(i) for i in line[1:]]

        return sensor_data[panel[:5]]

    def read_data_file():
        """
        Creates the unconverted vector and feature label list to be used later.
        """

        with open(path) as file:
            lines = file.readlines()
            feature_labels_all = lines[0].strip().split('\t')

            data_lists_left = []
            data_lists_right = []

            left_start, left_end, right_start, right_end = read_sensor_file()

            feature_labels_left = [i for i in feature_labels_all if left_end >= float(i) >= left_start]
            feature_labels_right = [i for i in feature_labels_all if right_end >= float(i) >= right_start]

            left_index_start = feature_labels_all.index(feature_labels_left[0])
            left_index_stop = feature_labels_all.index(feature_labels_left[-1])

            right_index_start = feature_labels_all.index(feature_labels_right[0])
            right_index_stop = feature_labels_all.index(feature_labels_right[-1])

            for line in lines[1:]:
                line_data = line.strip().split('\t')

                data_lists_left.append([line_data[0]] + line_data[left_index_start:left_index_stop + 1])
                data_lists_right.append([line_data[0]] + line_data[right_index_start:right_index_stop + 1])

            data_left = np.array(data_lists_left, dtype=object)
            data_right = np.array(data_lists_right, dtype=object)

            return data_left, data_right, feature_labels_left, feature_labels_right

    def convert_array(array):
        """
        Changes all dates to timestamps_clustered, NaN strings to NaN values and remaining strings to floats.
        """
        for i in range(len(array)):
            for j in range(len(array[i])):
                if j == 0:
                    year, month, rest = array[i, j].split('-')
                    day, rest = rest.split('T')
                    hour, minute, second = rest.split(':')

                    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute),
                                             int(np.floor(float(second))))
                    array[i, j] = datetime.datetime.timestamp(date)

                elif array[i, j] == 'NaN':
                    array[i, j] = np.nan
                else:
                    array[i, j] = float(array[i, j])

    array_left, array_right, labels_left, labels_right = read_data_file()

    convert_array(array_left)
    convert_array(array_right)

    return array_left, array_right, labels_left, labels_right


def gradient_arrays(array):
    """
    Returns tuple, first entry time derivative vector, second entry the length derivative vector.
    """
    time_derivative_array, length_derivative_array = np.gradient(array)

    for i in range(len(time_derivative_array)):
        for j in range(len(time_derivative_array[i])):
            if np.isnan(time_derivative_array[i, j]):
                time_derivative_array[i, j] = 0
            if np.isnan(length_derivative_array[i, j]):
                length_derivative_array[i, j] = 0

    return time_derivative_array, length_derivative_array


def array_to_image(array):
    """
    Generates a new vector with each value in the original vector converted to an RGB color.
    """
    min_value, max_value = np.nanmin(array) / 4, np.nanmax(array) / 4

    image = []

    for i in range(len(array)):
        image_row = []

        for j in range(len(array[i])):

            if array[i, j] <= 0:
                image_column = [max(1 - (array[i, j] / min_value), 0), max(1 - (array[i, j] / min_value), 0), 1]
            elif array[i, j] > 0:
                image_column = [1, max(1 - (array[i, j] / max_value), 0), max(1 - (array[i, j] / max_value), 0)]
            else:
                image_column = [0, 0, 0]

            image_row.append(image_column)

        image.append(image_row)

    return np.flip(image, axis=0)


def folder_to_array(panel, path):
    """
    Reads all files of a panel and converts them to left foot and right foot numpy arrays.
    """
    files_all = os.listdir(path)
    files_data = []

    for file in files_all:
        if file[:5] == panel:
            files_data.append(path + file)

    final_left_array = []
    final_right_array = []

    final_file_vector = []
    count = 0

    for file in files_data:
        left_array, right_array, _, _ = file_to_array(panel, file)

        if len(final_left_array) == 0:
            final_left_array, final_right_array = left_array, right_array
            final_file_vector = np.ones((len(left_array), 1)) * count
        else:
            final_left_array = np.vstack((final_left_array, left_array))
            final_right_array = np.vstack((final_right_array, right_array))

            file_vector = np.ones((len(left_array), 1)) * count
            final_file_vector = np.vstack((final_file_vector, file_vector))

        count += 1

    return final_left_array, final_right_array, final_file_vector


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


def preprocess_array(array, margin_start=20, margin_small=0.5, margin_medium=0.5, margin_big=0.1, length=18, plot=True):

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
