import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import datetime

import sklearn.cluster as cl
from sklearn import metrics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def raw_to_array_complete(folder_path, file_name):
    """
    Opens file in default LUNA data format and converts this into a numpy vector and pandas dataframe.
    """
    def read_file():
        """
        Creates the unconverted vector and feature label list to be used later for the dataframe.
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

    data_np, labels = read_file()
    convert_array(data_np)
    data_pd = pd.DataFrame(data_np, columns=labels)

    return data_np, data_pd, labels


def plot_image_complete(image):
    plt.imshow(image, extent=[0, 5000, 500, 0])
    plt.xlabel('L [mm]')
    plt.ylabel('timestamp [-]')
    plt.show()


def normalize_array(array):
    initial_conditions = array[0]
    min_value, max_value = - np.nanmin(array), np.nanmax(array)

    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i, j] = array[i, j] - initial_conditions[j]

            if array[i, j] <= 0:
                array[i, j] = array[i, j] / min_value
            elif array[i, j] > 0:
                array[i, j] = array[i, j] / max_value


def print_line_of_array(array, row_numbs, difference=False, color1="g", color2="r"):
    """
    vector = np.vector
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
        # row is a row from vector with length
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
        ax1.set_title(f"vector numb {row_numbs[0]}")
        ax2.plot(row2, f"{color1}o-", label=f"row {row_numb+1}")
        ax2.set_title(f"vector numb {row_numbs[1]}")
        ax3.plot(diffrow, f'{color2}')
        ax3.set_title(f"absolute difference")
    plt.show()


def cluster_to_image(vector):
    image = []

    for i in range(len(vector)):

        if vector[i, 0] == 0:
            image_row = [[1.0, 1.0, 1.0]]
        elif vector[i, 0] == 1:
            image_row = [[1.0, 0.0, 0.0]]
        else:
            image_row = [[0.0, 0.0, 1.0]]

        image.append(image_row)

    return np.flip(image, axis=0)


def do_in_batches(array, batches, function):
    i = 0
    output = None
    values_output = []
    for sub_array in np.array_split(array, batches, axis=0):
        begin_time = time.time()
        print(sub_array)
        cluster, values = function(sub_array) # meanshift(sub_array)
        values_output.append(values)
        if i == 0:
            output = cluster
        if i != 0:
            output = np.hstack((output, cluster))
        i += 1
        print(f'runtime for batch {i} = {begin_time - time.time()}')
    return output, values_output

def cluster_to_image(cluster):
    cluster_image = np.ones(cluster.shape)

    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            if not cluster[i, j]:
                cluster_image[i, j] = 0.0

    return np.flip(cluster_image, axis=0)


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
    return


def mean_shift(array, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.MeanShift()
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def aff_prop(array, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.AffinityPropagation()
    cluster = model.fit_predict(array)
    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def agglo(array, n=5, scaled=True):
    # scale array
    if scaled:
        array = scaling(array)

    # clustering
    model = cl.AgglomerativeClustering(n_clusters=n, distance_threshold=1)
    cluster = model.fit_predict(array)

    cluster_values = np.unique(cluster)

    # output
    return cluster.reshape(-1, 1), cluster_values


def silhouette_score(array, labels):  # if needed could be called independent
    score = metrics.silhouette_score(array, labels)
    return score


def calinski_score(array, labels):  # if needed could be called independent
    score = metrics.calinski_harabasz_score(array, labels)
    return score


def davies_score(array, labels):  # if needed could be called independent
    score = metrics.davies_bouldin_score(array, labels)
    return score


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
                # image_column = [max(1 - (array[i, j] / min_value), 0), max(1 - (array[i, j] / min_value), 0), 1]

                image_column = - max(1 - (array[i, j] / min_value), 0)

            elif array[i, j] > 0:
                # image_column = [1, max(1 - (array[i, j] / max_value), 0), max(1 - (array[i, j] / max_value), 0)]

                image_column = max(1 - (array[i, j] / min_value), 0)

            else:
                # image_column = [0, 0, 0]

                image_column = 0

            image_row.append(image_column)

        image.append(image_row)

    # return np.flip(np.transpose(image, (1, 0, 2)), axis=0)

    return np.flip(np.transpose(image, (1, 0, 2)), axis=0)


def plot_arrays(image, image_time, image_length, length, time, panel, left=True):
    """
    Plots the original vector, time derivative vector and length derivative vector including a color bar.
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
        plt.suptitle(f'Left foot panel {panel}')
    else:
        plt.suptitle(f'Right foot panel {panel}')

    cbr = plt.colorbar(plt.cm.ScalarMappable(cmap='bwr', norm=mpl.Normalize(vmin=-1, vmax=1)))
    cbr.set_label('Scaled values [-]')
    plt.show()


def plot_cluster(im_time_left, im_time_right, im_length_left, im_length_right, im_cluster_left, im_cluster_right,
                 length_left, length_right, time, panel):

    figure = plt.figure(constrained_layout=True)
    figure.supxlabel('Length measurements [-]')
    figure.supylabel('Time measurements [-]')
    figure.suptitle(f'Panel {panel}')

    sub_figures = figure.subfigures(1, 3)
    sub_figures[0].suptitle('Time derivatives')
    sub_figures[1].suptitle('Length derivatives')
    sub_figures[2].suptitle('Clusters')

    axs0 = sub_figures[0].subplots(1, 2, sharey=True)
    axs0[0].imshow(im_time_left, extent=[0, length_left, 0, time], aspect='auto')
    axs0[0].set_title('Left')

    axs0[1].imshow(im_time_right, extent=[0, length_right, 0, time], aspect='auto')
    axs0[1].set_title('Right')

    axs1 = sub_figures[1].subplots(1, 2, sharey=True)
    axs1[0].imshow(im_length_left, extent=[0, length_left, 0, time], aspect='auto')
    axs1[0].set_title('Left')

    axs1[1].imshow(im_length_right, extent=[0, length_right, 0, time], aspect='auto')
    axs1[1].set_title('Right')

    axs2 = sub_figures[2].subplots(1, 2, sharey=True)
    axs2[0].imshow(im_cluster_left, extent=[0, length_left, 0, time], aspect='auto', cmap='gray')
    axs2[0].set_title('Left')

    axs2[1].imshow(im_cluster_right, extent=[0, length_right, 0, time], aspect='auto', cmap='gray')
    axs2[1].set_title('Right')

    plt.show()


def plot_example_cluster(image_time, cluster_array, cluster_name, cluster_values, length, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Reference')

    image_cluster = np.flip(cluster_array, axis=0)

    plt.subplot(1, 2, 2)
    plt.imshow(image_cluster, extent=[0, length, 0, time], cmap='inferno')
    plt.xlabel(f'Number of clusters: {len(cluster_values)}')
    plt.ylabel('Timestamp [-]')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{cluster_name} clustering')

    plt.show()


def plot_clusters(image_left, image_right, length_left_labels, length_right_labels, time_labels, panel, division=20):

    labels_time = []
    labels_length_left = [int(float(length_left_labels[0])), int(float(length_left_labels[len(length_left_labels) // 2])), int(float(length_left_labels[-1]))]
    labels_length_right = [int(float(length_right_labels[0])),
                           int(float(length_right_labels[len(length_left_labels) // 2])),
                           int(float(length_right_labels[-1]))]

    for i in range(len(time_labels)):
        if i % division == 0 and i == 0:
            labels_time.append(0)

        elif i % division == 0:
            labels_time.append(int(float(time_labels[i])))

    figure = plt.figure(constrained_layout=True)
    # figure.supxlabel('Length measurements [-]')
    # figure.supylabel('Time measurements [-]')
    figure.suptitle(f'Panel {panel}')

    sub_figures = figure.subfigures(2, 1)
    sub_figures[0].suptitle('LUNA clusters')
    sub_figures[1].suptitle('AE clusters')

    axs0 = sub_figures[0].subplots(2, 1, sharex=True)
    axs0[0].imshow(image_left, extent=[0, len(time_labels), 0, len(length_left_labels)], aspect='auto')
    # axs0[0].set_title('Left')

    axs0[0].set_xticks(np.arange(len(labels_time)) * division)
    axs0[0].set_xticklabels(labels_time)

    axs0[0].set_yticks([0, 0.5 * len(length_left_labels), len(length_left_labels)])
    axs0[0].set_yticklabels(labels_length_left)
    axs0[0].set_ylabel('length [mm]')

    axs0[1].imshow(image_right, extent=[0, len(time_labels), 0, len(length_right_labels)], aspect='auto')
    # axs0[1].set_title('Right')
    axs0[1].set_xlabel('time [s]')

    axs0[1].set_yticks([0, 0.5 * len(length_right_labels), len(length_right_labels)])
    axs0[1].set_yticklabels(labels_length_right)
    axs0[1].set_ylabel('length [mm]')
    axs0[1].set_ylabel('length [mm]')

    plt.show()