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


def plot_image_complete(image):
    plt.imshow(image, extent=[0, 5000, 500, 0])
    plt.xlabel('L [mm]')
    plt.ylabel('timestamp [-]')
    plt.show()


def print_line_of_array(array, row_numbs, difference=False, color1="g", color2="r"):
    """
    array = np.array
    row_numbs is a list, could be single value list
    difference is to plot absolute difference between two rows. Only works if row_numbs is 2 value list
    color1 is color, dafault green
    color2 is color, default red
    """
    if difference is True and len(row_numbs)!= 2:
        print("Function Error: Change difference to False or use a 2 value list")
        return

    while difference and len(row_numbs) == 2:
        diffrow = []
        row1 = array[row_numbs[0]]
        row2 = array[row_numbs[1]]
        for j in range(len(row1)):
            diffrow.append(abs(row1[j]) - abs(row2[j])) # absolute difference
        break

    for row_numb in row_numbs:
        row = array[row_numb]
        # row is a row from array with length
        rowlengt = len(row)
        avg_row_value = np.nansum(row)/rowlengt
        maxvalue, minvalue = np.nanmax(row), np.nanmin(row)
        print(avg_row_value, maxvalue, minvalue)
        if difference is False:
            plt.plot(row, "o-", label=f"row {row_numb+1}")
            plt.title("Lengt x Micro strain")
            plt.xlabel("Length")
            plt.ylabel("Micro strain")
            plt.legend()

    if difference is True:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.plot(row1, f"{color1}o-", label=f"row {row_numb+1}")
        ax1.set_title(f"array numb {row_numbs[0]}")
        ax2.plot(row2, f"{color1}o-", label=f"row {row_numb+1}")
        ax2.set_title(f"array numb {row_numbs[1]}")
        ax3.plot(diffrow, f'{color2}')
        ax3.set_title(f"absolute difference")
    plt.show()
