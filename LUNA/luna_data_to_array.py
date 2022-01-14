import numpy as np

import datetime
import os


def file_to_array(panel, path):
    """
    Opens file in default LUNA data format and converts this into left and right foot arrays.
    """
    def read_sensor_file():
        """
        Reads the sensor file and returns the relevant data for the specified panel.
        """
        sensor_file = '/LUNA_sensor.txt'
        sensor_data = {}

        with open(os.path.dirname(__file__) + sensor_file) as file:
            data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype=str)

            for line in data:
                sensor_data[line[0]] = [float(i) for i in line[1:]]

        return sensor_data[panel[:5]]

    def read_data_file():
        """
        Creates the unconverted array and feature label list to be used later.
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
        Changes all dates to timestamps, NaN strings to NaN values and remaining strings to floats.
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
    Returns tuple, first entry time derivative array, second entry the length derivative array.
    """
    time_derivative_array, length_derivative_array = np.gradient(array)

    for i in range(len(time_derivative_array)):
        for j in range(len(time_derivative_array[i])):
            if np.isnan(time_derivative_array[i, j]):
                time_derivative_array[i, j] = 0
            if np.isnan(length_derivative_array[i, j]):
                length_derivative_array[i, j] = 0

    return time_derivative_array, length_derivative_array


def folder_to_array(panel, path):
    """
    Reads all files of a panel and converts them to left foot and right foot arrays.
    """
    files_all = os.listdir(path)
    files_data = []

    # Put all files in 1 list
    for file in files_all:
        if file[:5] == panel:
            files_data.append(path + file)

    # Create empty arrays to fill with data and setting the file counter to 0
    final_left_array, final_right_array = [], []
    final_file_vector = []
    left_labels, right_labels = [], []
    count = 0

    # Go through all files and put the data in the final arrays using the file_to_array function
    for file in files_data:
        left_array, right_array, left_labels, right_labels = file_to_array(panel, file)

        if len(final_left_array) == 0:
            final_left_array, final_right_array = left_array, right_array
            final_file_vector = np.ones((len(left_array), 1)) * count
        else:
            final_left_array = np.vstack((final_left_array, left_array))
            final_right_array = np.vstack((final_right_array, right_array))

            file_vector = np.ones((len(left_array), 1)) * count
            final_file_vector = np.vstack((final_file_vector, file_vector))

        count += 1

    return final_left_array, final_right_array, final_file_vector, left_labels, right_labels
