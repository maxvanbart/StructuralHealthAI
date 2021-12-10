import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def raw_to_array(folder_path, file_name):
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


def array_to_image(array):
    def convert_array():
        min_value, max_value = np.nanmin(array), np.nanmax(array)

        print(min_value, max_value)

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

    def plot_image(image):
        plt.imshow(image)
        plt.xlabel('L [mm]')
        plt.ylabel('t [s]')
        plt.show()

    image = convert_array()
    plot_image(image)

    return image


def demo():
    folder = 'Files/L1-03/LUNA/'
    file = 'L1-03.txt'

    array, dataframe, labels = raw_to_array(folder, file)
    image = array_to_image(array[:, 1:])


if __name__ == '__main__':
    demo()
