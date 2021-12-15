import numpy as np

import datetime


def raw_to_array(panel):
    """
    Opens file in default LUNA data format and converts this into left and right foot numpy arrays.
    """
    def read_sensor_file():
        sensor_file = 'LUNA_sensor.txt'
        sensor_data = {}

        with open(sensor_file) as file:
            data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype=str)

            for line in data:
                sensor_data[line[0]] = [float(i) for i in line[1:]]

        return sensor_data[panel[:5]]

    def read_data_file():
        """
        Creates the unconverted vector and feature label list to be used later.
        """
        path = f'Files/{panel[:5]}/LUNA/{panel}.txt'

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

            array_left = np.array(data_lists_left, dtype=object)
            array_right = np.array(data_lists_right, dtype=object)

            return array_left, array_right, feature_labels_left, feature_labels_right

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

    data_np_left, data_np_right, labels_left, labels_right = read_data_file()

    convert_array(data_np_left)
    convert_array(data_np_right)

    return data_np_left, data_np_right, labels_left, labels_right


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