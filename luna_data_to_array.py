import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def raw_to_array_complete(folder_path, file_name):
    """
    Opens file in default LUNA data format and converts this into a numpy array and pandas dataframe.
    """
    def read_file():
        """
        Creates the unconverted array and feature label list to be used later for the dataframe.
        """
        with open(folder_path + file_name) as file:
            lines = file.readlines()
            feature_labels = lines[0].strip().split('\t')
            feature_labels.insert(0, 'timestamp')
            data_lists = []

            for line in lines[1:]:
                data_lists.append(line.strip().split('\t'))

            return np.array(data_lists, dtype=object), feature_labels

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

    data_np, labels = read_file()
    convert_array(data_np)
    data_pd = pd.DataFrame(data_np, columns=labels)

    return data_np, data_pd, labels


def raw_to_array(folder_path, file_name, left_start, left_end, right_start, right_end):
    """
    Opens file in default LUNA data format and converts this into two numpy arrays and two pandas dataframes.
    """

    def read_file():
        """
        Creates the unconverted array and feature label list to be used later for the dataframe.
        """
        with open(folder_path + file_name) as file:
            lines = file.readlines()
            feature_labels_all = lines[0].strip().split('\t')

            data_lists_left = []
            data_lists_right = []

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

            return array_left, array_right, ['t'] + feature_labels_left, ['t'] + feature_labels_right

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

    data_np_left, data_np_right, labels_left, labels_right = read_file()

    convert_array(data_np_left)
    convert_array(data_np_right)

    data_pd_left = pd.DataFrame(data_np_left, columns=labels_left)
    data_pd_right = pd.DataFrame(data_np_right, columns=labels_right)

    return data_np_left, data_np_right, data_pd_left, data_pd_right, labels_left, labels_right


def array_to_image(array):
    min_value, max_value = np.nanmin(array) / 32, np.nanmax(array) / 32

    image = []

    for i in range(len(array)):
        image_row = []

        for j in range(len(array[i])):

            if array[i, j] <= 0:
                image_column = [1 - (array[i, j] / min_value), 1, 1]
            elif array[i, j] > 0:
                image_column = [1, 1 - (array[i, j] / max_value), 1]
            else:
                image_column = [0, 0, 0]

            image_row.append(image_column)

        image.append(image_row)

    return image


def gradient_array(array):
    deriv_t, deriv_x = np.gradient(array)

    return deriv_t, deriv_x


def plot_image_complete(image):
    plt.imshow(image, extent=[0, 5000, 500, 0])
    plt.xlabel('L [mm]')
    plt.ylabel('timestamp [-]')
    plt.show()


def plot_images(image_left, image_right, length_left, length_right, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_left, extent=[0, length_left, time, 0])
    plt.xlabel('L [mm]')
    plt.ylabel('timestamp [-]')
    plt.title('Left foot')

    plt.subplot(1, 2, 2)
    plt.imshow(image_right, extent=[0, length_right, time, 0])
    plt.xlabel('L [mm]')
    plt.ylabel('timestamp [-]')
    plt.title('Right foot')

    plt.show()


def demo_complete():
    # folder = 'Files/L1-03/LUNA/'
    # file = 'L1-03.txt'

    folder = 'Files/L1-04/LUNA/'
    file = 'L1-04-2.txt'

    array, dataframe, labels = raw_to_array_complete(folder, file)
    image = array_to_image(array)
    plot_image_complete(image)


def demo_left_right():
    # folder = 'Files/L1-03/LUNA/'
    # file = 'L1-03.txt'

    folder = 'Files/L1-23/'
    file = 'L1-23-3.txt'

    left_start, left_end = 845, 1045
    right_start, right_end = 2770, 2920

    delta_length_left = left_end - left_start
    delta_length_right = right_end - right_start

    array_left, array_right, dataframe_left, dataframe_right, labels_left, labels_right = \
        raw_to_array(folder, file, left_start, left_end, right_start, right_end)

    deriv_t, deriv_x = gradient_array(array_right[:, 1:])

    deriv_t_image = array_to_image(deriv_t)
    deriv_x_image = array_to_image(deriv_x)

    plot_images(deriv_t_image, deriv_x_image, 130, 180, 562)

    image_left = array_to_image(array_left[:, 1:])
    image_right = array_to_image(array_right[:, 1:])

    plot_images(image_left, image_right, delta_length_left, delta_length_right, len(image_left))


if __name__ == '__main__':
    demo_left_right()
