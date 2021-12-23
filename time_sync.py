import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from LUNA.luna_data_to_array import file_to_array


def package_databases(data_ae_np, data_luna_np, timestamps_AE, timestamps_LUNA):

    final_array = [[]]
    row = 0

    # Getting the intervals from LUNA.
    intervals_np = [timestamps_LUNA[i + 1] - timestamps_LUNA[i] for i in range(len(timestamps_LUNA) - 1)]
    intervals_pd = pd.DataFrame(intervals_np, dtype=float)
    print(intervals_np)
    intervals_counts = [i[0] for i in intervals_pd.value_counts().index.tolist()]
    intervals_big = np.max(intervals_counts[:2])
    intervals_small = np.min(intervals_counts[:2])

    # loop over all LUNA timestamps
    for i in range(len(timestamps_LUNA) - 1):

        # check if LUNA timestamp is the first one of a big batch with 10 smaller ribbons.
        if intervals_np[i] < intervals_small + 20 and intervals_np[i] > intervals_small - 20:
            if i == 0 or intervals_np[i - 1] > 2 * intervals_big:
                dict = {}
                dict['LUNA_end'] = data_luna_np[i]

                # collect all corresponding AE data
                ae_lst = []
                for j in range(len(timestamps_AE)):
                    if timestamps_AE[j] < timestamps_LUNA[i] and timestamps_LUNA[i] - timestamps_AE[j] <200:
                        ae_lst.append(data_ae_np[j])

                dict['AE'] = ae_lst

                # put dictionary in final array
                final_array[row].append(dict)

        # Check if LUNA timestamp is a 'LUNA start' point
        elif intervals_np[i] < intervals_big + 20 and intervals_np[i] > intervals_big - 20:
            dict = {}
            dict['LUNA_start'] = data_luna_np[i]
            dict['LUNA_end'] = data_luna_np[i+1]

            # collect all corresponding AE data
            ae_lst = []
            for j in range(len(timestamps_AE)):
                if timestamps_AE[j] < timestamps_LUNA[i+1] and timestamps_AE[j] > timestamps_LUNA[i]:
                    ae_lst.append(data_ae_np[j])

            dict['AE'] = ae_lst

            # put dictionary in final array
            final_array[row].append(dict)

        # Check if LUNA timestamp is the last point of a big batch with 10 smaller ribbons.
        elif intervals_np[i] > 2 * intervals_big:
            dict = {}
            dict['LUNA_start'] = data_luna_np[i]

            # collect all corresponding AE data
            ae_lst = []
            for j in range(len(timestamps_AE)):
                if timestamps_AE[j] > timestamps_LUNA[i] and timestamps_AE[j] - timestamps_LUNA[i] < 200:
                    ae_lst.append(data_ae_np[j])

            dict['AE'] = ae_lst

            # put dictionary in final array
            final_array[row].append(dict)
            final_array.append([])
            row += 1

    return final_array


def synchronize_databases(array_ae, array_luna, samples=100, margin_start=0, margin_end=10):

    def remove_outliers_luna():

        def remove_outliers_end():
            cut_end = -1

            for i in range(len(timestamps)):
                if timestamps[-i] - timestamps[-i - 1] > intervals_mean:
                    cut_end = -i

            return timestamps[:cut_end], database[:cut_end]

        def remove_outliers_start():
            cut_start = 0

            for i in range(len(timestamps)):
                if intervals_big + 10 > intervals_np[i] > intervals_big - 10:
                    cut_start = i
                    break
                elif intervals_small + 10 > intervals_np[i] > intervals_small - 10:
                    cut_start = i
                    break

            return timestamps[cut_start:], cut_start

        def remove_outliers_middle(length=18):
            intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            intervals.insert(0, 0)

            indices_to_be_removed = []

            for i in range(len(intervals)):

                if i % 2 == 0 and i != 0 and i % length != 0:
                    if intervals[i] < intervals_big - 20 or intervals[i] > intervals_big + 20:
                        indices_to_be_removed.append(i)

                elif i % 2 == 1:
                    if intervals[i] < intervals_small - 20 or intervals[i] > intervals_small + 20:
                        indices_to_be_removed.append(i)

            if not indices_to_be_removed:
                return timestamps, True
            else:
                return np.delete(timestamps, indices_to_be_removed[0]), \
                       np.delete(database, indices_to_be_removed[0], 0), False

        timestamps = array_luna[:, 0] - array_luna[0, 0]
        database = array_luna
        timestamps, database = remove_outliers_end()
        timestamps, database = remove_outliers_start()
        timestamps, database, completed = remove_outliers_middle()

        while not completed:
            timestamps, database, completed = remove_outliers_middle()

        return timestamps, database

    # 0. gathering information.
    # Getting timestamps from arrays.
    timestamps_luna = array_luna[:, 0] - array_luna[0, 0]
    timestamps_ae = array_ae[:, 0]

    values_ae_uncut = array_ae[:, 4]
    values_luna_uncut = np.ones(len(timestamps_luna)) * np.mean(values_ae_uncut)

    # Getting the intervals from LUNA.
    intervals_np = [timestamps_luna[i + 1] - timestamps_luna[i] for i in range(len(timestamps_luna) - 1)]
    intervals_pd = pd.DataFrame(intervals_np, dtype=float)

    intervals_counts = [i[0] for i in intervals_pd.value_counts().index.tolist()]
    intervals_big = np.max(intervals_counts[:2])
    intervals_small = np.min(intervals_counts[:2])
    intervals_mean = np.std(intervals_np)

    timestamps_luna, database_luna = remove_outliers_luna()

    # 2. translating LUNA and AE to synchronize start at zero time.
    translation_ae = np.mean(timestamps_ae[:samples])
    translation_luna = timestamps_luna[0] - intervals_big

    timestamps_ae = timestamps_ae - translation_ae
    timestamps_luna = timestamps_luna - translation_luna

    # 3. removing outliers from AE data.
    # Cutting outliers from AE start.
    cut_ae_start = 0

    for i in range(len(timestamps_ae)):
        if timestamps_ae[i] < margin_start:
            cut_ae_start = i

    # Cutting outliers from AE end.
    cut_ae_end = -1

    for i in range(len(timestamps_ae)):
        if timestamps_ae[-i] > timestamps_luna[-1] + intervals_big + margin_end:
            cut_ae_end = -i

    timestamps_ae = timestamps_ae[cut_ae_start: cut_ae_end]

    # 4. sanity check if data is correctly aligned.
    mean_ae, std_ae = np.mean(values_ae_uncut), np.std(values_ae_uncut)

    bar = mean_ae - 1 * std_ae

    values_ae = np.array([1 if value_ae > bar else 0 for value_ae in values_ae_uncut[cut_ae_start: cut_ae_end]])

    sample_y_values_LUNA = np.ones((len(timestamps_luna)))
    sample_y_values_AE = values_ae

    length_set = 10 * intervals_big + 9 * intervals_small

    with open('reference_ae.txt') as file:
        reference_array_ae = np.genfromtxt(file)
    with open('reference_luna.txt') as file:
        reference_array_luna = np.genfromtxt(file)

    plt.subplot(2, 1, 1)
    plt.scatter(timestamps_ae, sample_y_values_AE)
    plt.scatter(timestamps_luna, sample_y_values_LUNA, s=100)
    plt.xlim(-50, length_set + 50)
    plt.ylim(0.99, 1.01)
    plt.title('Synced panel')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 1, 2)
    plt.scatter(reference_array_ae[:, 0], reference_array_ae[:, 1])
    plt.scatter(reference_array_luna[:, 0], reference_array_luna[:, 1], s=100)
    plt.xlim(-50, length_set + 50)
    plt.ylim(0.99, 1.01)
    plt.title('Correctly synced')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    values_ae = array_ae[:, 4]
    values_luna = np.ones(len(timestamps_luna)) * np.mean(values_ae)

    # complete plot

    plt.scatter(timestamps_ae, values_ae)
    plt.scatter(timestamps_luna, values_luna)
    plt.show()

    return


# opening the files
panel = 'L1-09'
path_luna = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{panel}.txt'
path_ae = os.path.dirname(__file__) + f'/Files/{panel}/AE/{panel}.csv'

# to arrays
data_ae_pd_unsorted = pd.read_csv(path_ae)
data_ae_np_unsorted = data_ae_pd_unsorted.to_numpy(dtype=float)

data_ae_np = data_ae_np_unsorted[np.argsort(data_ae_np_unsorted[:, 0])]
data_luna_np, _, _, _ = file_to_array(panel, path_luna)

synchronize_databases(data_ae_np, data_luna_np)
