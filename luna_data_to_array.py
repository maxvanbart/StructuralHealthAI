import numpy as np
import matplotlib.colors as mpl
import matplotlib.pyplot as plt

import datetime


def raw_to_array(folder_path, file_name, panel):
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

        return sensor_data[panel]

    def read_data_file():
        """
        Creates the unconverted array and feature label list to be used later.
        """
        with open(folder_path + file_name) as file:
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


def array_to_image(array):
    """
    Generates a new array with each value in the original array converted to an RGB color.
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


def plot_arrays(image, image_time, image_length, length, time, left=True):
    """
    Plots the original array, time derivative array and length derivative array including a color bar.
    """
    plt.subplot(1, 3, 1)
    plt.imshow(image, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Micro strain')

    plt.subplot(1, 3, 2)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Time (partial) derivative')

    plt.subplot(1, 3, 3)
    plt.imshow(image_length, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Length (partial) derivative')

    if left:
        plt.suptitle('Left foot panel L1-03')
    else:
        plt.suptitle('Right foot panel L1-03')

    cbr = plt.colorbar(plt.cm.ScalarMappable(cmap='bwr', norm=mpl.Normalize(vmin=-1, vmax=1)))
    cbr.set_label('Scaled values [-]')
    plt.show()


def plot_cluster(image_time, image_cluster, length, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Reference')

    plt.subplot(1, 2, 2)
    plt.imshow(image_cluster, extent=[0, length, 0, time])
    plt.xticks([])
    plt.yticks([])
    plt.title('Cluster labels')

    plt.show()


def demo():

    # --- USER INPUT ---
    folder = 'Files/L1-05/LUNA/'
    file = 'L1-05-2.txt'
    panel = 'L1-05'
    # ------------------

    array_left, array_right, labels_left, labels_right = raw_to_array(folder, file, panel)

    delta_length_left = float(labels_left[-1]) - float(labels_left[0])
    delta_length_right = float(labels_right[-1]) - float(labels_right[0])

    delta_time_left = len(array_left)
    delta_time_right = len(array_right)

    time_derivative_array_left, length_derivative_array_left = gradient_arrays(array_left[:, 1:])
    time_derivative_array_right, length_derivative_array_right = gradient_arrays(array_right[:, 1:])

    image_left = array_to_image(array_left)
    image_right = array_to_image(array_right)

    time_derivative_image_left = array_to_image(time_derivative_array_left)
    length_derivative_image_left = array_to_image(length_derivative_array_left)

    time_derivative_image_right = array_to_image(time_derivative_array_right)
    length_derivative_image_right = array_to_image(length_derivative_array_right)

    plot_cluster(time_derivative_image_right, time_derivative_image_right, delta_length_right, delta_time_right)

    plot_arrays(image_left, time_derivative_image_left, length_derivative_image_left,
                delta_length_left, delta_time_left)

    plot_arrays(image_right, time_derivative_image_right, length_derivative_image_right,
                delta_length_right, delta_time_right, left=False)


if __name__ == '__main__':
    demo()
