import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import AgglomerativeClustering

from LUNA.luna_data_to_array import file_to_array, folder_to_array


# Test file, complete chaos!
# Test file, complete chaos!


def package_databases(data_ae_np, data_luna_np, timestamps_ae, timestamps_luna):

    # get timestamps_luna_clustered
    # timestamps_ae = data_ae_np[:,0]
    # data_ae_np = data_ae_np[:,1:]
    # timestamps_luna = data_luna_np[:,0]
    # data_luna_np = data_luna_np[:,1:]

    final_array = [[]]
    row = 0

    # Getting the intervals from LUNA.
    intervals_np = [timestamps_luna[i + 1] - timestamps_luna[i] for i in range(len(timestamps_luna) - 1)]
    intervals_pd = pd.DataFrame(intervals_np, dtype=float)
    print(intervals_np)
    intervals_counts = [i[0] for i in intervals_pd.value_counts().index.tolist()]
    intervals_big = np.max(intervals_counts[:2])
    intervals_small = np.min(intervals_counts[:2])

    # loop over all LUNA timestamps_luna_clustered
    for i in range(len(timestamps_luna) - 1):

        # check if LUNA timestamp is the first one of a big batch with 10 smaller ribbons.
        if intervals_small - 20 < intervals_np[i] < intervals_small + 20:
            if i == 0 or intervals_np[i - 1] > 2 * intervals_big:
                dict = {}
                dict['LUNA_end'] = data_luna_np[i]

                # collect all corresponding AE data
                ae_lst = []
                for j in range(len(timestamps_ae)):
                    if timestamps_ae[j] < timestamps_luna[i] and timestamps_luna[i] - timestamps_ae[j] < 200:
                        ae_lst.append(data_ae_np[j])

                dict['AE'] = ae_lst

                # put dictionary in final array
                final_array[row].append(dict)

        # Check if LUNA timestamp is a 'LUNA start' point
        elif intervals_big - 20 < intervals_np[i] < intervals_big + 20:
            dict = {}
            dict['LUNA_start'] = data_luna_np[i]
            dict['LUNA_end'] = data_luna_np[i+1]

            # collect all corresponding AE data
            ae_lst = []
            for j in range(len(timestamps_ae)):
                if timestamps_luna[i] < timestamps_ae[j] < timestamps_luna[i+1]:
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
            for j in range(len(timestamps_ae)):
                if timestamps_ae[j] > timestamps_luna[i] and timestamps_ae[j] - timestamps_luna[i] < 200:
                    ae_lst.append(data_ae_np[j])

            dict['AE'] = ae_lst

            # put dictionary in final array
            final_array[row].append(dict)
            final_array.append([])
            row += 1

    return final_array


def synchronize_databases(array_ae, array_luna, margin_ae=100, margin_luna=20, length=18, multiple=False):

    def cluster_luna():
        pass

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
                if intervals_big + margin_luna > intervals_np[i] > intervals_big - margin_luna:
                    cut_start = i
                    break
                elif intervals_small + margin_luna > intervals_np[i] > intervals_small - margin_luna:
                    cut_start = i
                    break

            return timestamps[cut_start:], database[cut_start:]

        def remove_outliers_middle():
            intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            intervals.insert(0, 0)

            index_to_be_removed = None

            for i in range(len(intervals)):

                if i % 2 == 0 and i != 0 and i % length != 0:
                    if intervals[i] < intervals_big - margin_luna or intervals[i] > intervals_big + margin_luna:
                        index_to_be_removed = i
                        break

                elif i % 2 == 1:
                    if intervals[i] < intervals_small - margin_luna or intervals[i] > intervals_small + margin_luna:
                        index_to_be_removed = i
                        break

            if index_to_be_removed is None:
                return timestamps, database, True
            else:
                return np.delete(timestamps, index_to_be_removed), \
                       np.delete(database, index_to_be_removed, 0), False

        timestamps = array_luna[:, 0] - array_luna[0, 0]
        database = array_luna
        timestamps, database = remove_outliers_end()
        timestamps, database = remove_outliers_start()
        timestamps, database, completed = remove_outliers_middle()

        while not completed:
            timestamps, database, completed = remove_outliers_middle()

        return timestamps, database

    def synchronization():
        pass

    def remove_outliers_ae():

        def remove_outliers_start():
            pass

        def remove_outliers_end():
            pass

        return

    def sanity_check():
        pass

    # 0. gathering information.
    # Getting timestamps_luna_clustered from arrays.
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
    translation_ae = np.mean(timestamps_ae[:margin_ae])
    translation_luna = timestamps_luna[0] - intervals_big

    timestamps_ae = timestamps_ae - translation_ae
    timestamps_luna = timestamps_luna - translation_luna

    # 3. removing outliers from AE data.
    # Cutting outliers from AE start.
    cut_ae_start = 0

    for i in range(len(timestamps_ae)):
        if timestamps_ae[i] < 0 - margin_luna:
            cut_ae_start = i

    # Cutting outliers from AE end.
    cut_ae_end = -1

    for i in range(len(timestamps_ae)):
        if timestamps_ae[-i] > timestamps_luna[-1] + intervals_big + margin_luna:
            cut_ae_end = -i

    timestamps_ae = timestamps_ae[cut_ae_start: cut_ae_end]

    # 4. sanity check if data is correctly aligned.
    mean_ae, std_ae = np.mean(values_ae_uncut), np.std(values_ae_uncut)

    bar = mean_ae - 2 * std_ae

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

    timestamps_ae = array_ae[:, 0]

    values_ae = array_ae[:, 4]
    values_luna = np.ones(len(timestamps_luna)) * np.mean(values_ae)

    # complete plot

    plt.scatter(timestamps_ae, values_ae)
    plt.scatter(timestamps_luna, values_luna)
    plt.show()

    return


def cluster_luna(timestamps):

    clustering = AgglomerativeClustering(n_clusters=None, linkage='single',
                                         distance_threshold=3000).fit(timestamps.reshape(-1, 1))
    clustered_timestamps = clustering.labels_

    values = np.ones(len(timestamps))

    # plt.scatter(timestamps, values, c=clustered_timestamps)
    # plt.show()

    return clustered_timestamps


def split_luna(timestamps_luna, timestamps_luna_clustered, database_luna, database_ae, length=1, margin_start=5, margin_end=10):
    timestamps_split = []
    database_luna_split = []
    database_ae_split = []

    database_luna[:, 0] = database_luna[:, 0] - database_luna[0, 0]

    count = 0
    previous_split = 0

    for i in range(len(timestamps_luna_clustered) - 1):
        count += 1

        if timestamps_luna_clustered[i + 1] != timestamps_luna_clustered[i] and count >= length:
            timestamps_split.append(timestamps[previous_split: i + 1])
            database_luna_split.append(database_luna[previous_split: i + 1, :])
            count = 0
            previous_split = i + 1

    database_luna_split.append(database_luna[previous_split: -1, :])
    timestamps_split.append(timestamps[previous_split: -1])

    previous_split = 0

    timestamps_ae = database_ae[:, 0]

    for timestamp in timestamps_split:
        start, end = timestamp[0], timestamp[-1]
        cut_start, cut_end = 0, 0
        sub_database = []

        for i in range(previous_split, len(timestamps_ae)):
            if timestamps_ae[i] < start - margin_start:
                cut_start = i + 1
            elif timestamps_ae[i] > end + margin_end:
                cut_end = i + 1
                database_ae_split.append(database_ae[cut_start: cut_end, :])
                break

    return timestamps_split, np.array(database_luna_split), np.array(database_ae_split)


def generate_test_data():

    timestamps_ae = []
    timestamps_luna = []

    data_ae = []
    data_luna = []

    small_step = 100
    big_step = 200
    small_gap = 1000
    large_gaps = [10000, 15000, 20000]

    # starting values
    luna_timestamp = 100
    ae_timestamp = 5

    for i in range(3):
        for j in range(5):
            for z in range(9):
                for k in range(10):
                    timestamps_ae.append(ae_timestamp)
                    data_ae.append(np.random.normal())
                    ae_timestamp += 20

                luna_timestamp += big_step
                timestamps_luna.append(luna_timestamp)
                data_luna.append(np.random.normal())

                ae_timestamp += small_step
                luna_timestamp += small_step
                timestamps_luna.append(luna_timestamp)
                data_luna.append(np.random.normal())

            ae_timestamp += small_gap
            luna_timestamp += small_gap

        ae_timestamp += large_gaps[i]
        luna_timestamp += large_gaps[i]

    timestamps_ae = np.array(timestamps_ae).reshape(-1, 1)
    timestamps_luna = np.array(timestamps_luna).reshape(-1, 1)
    data_ae = np.array(data_ae).reshape(-1, 1)
    data_luna = np.array(data_luna).reshape(-1, 1)

    ae_test_data = np.hstack((timestamps_ae, data_ae))
    luna_test_data = np.hstack((timestamps_luna, data_luna))

    values_luna = np.ones(len(data_luna))

    # plotiing
    # plt.scatter(timestamps_ae, data_ae)
    # plt.scatter(timestamps_luna, values_luna)
    # plt.show()

    return ae_test_data, luna_test_data


# # opening the files
# panel = 'L1-23'
# path_luna = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{panel}.txt'
# path_ae = os.path.dirname(__file__) + f'/Files/{panel}/AE/{panel}.csv'
#
# # to arrays
# data_ae_pd_unsorted = pd.read_csv(path_ae)
# data_ae_np_unsorted = data_ae_pd_unsorted.to_numpy(dtype=float)
#
# data_ae_np = data_ae_np_unsorted[np.argsort(data_ae_np_unsorted[:, 0])]
# data_luna_np, _, _, _ = file_to_array(panel, path_luna)
#
# # in case 2 files for LUNA
#
# folder_path = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/'
#
# data_luna_np, _ = folder_to_array(panel, folder_path)
#
# timestamps = data_luna_np[:, 0] - data_luna_np[0, 0]

data_ae_np, data_luna_np = generate_test_data()

timestamps = data_luna_np[:, 0] - data_luna_np[0, 0]

timestamps_clustered = cluster_luna(timestamps)
timestamps_split, database_luna, database_ae = split_luna(timestamps, timestamps_clustered, data_luna_np, data_ae_np)

print(database_luna[0][:5, 0])
print()
print(database_ae[0][:20, 0])

# synchronize_databases(data_ae_np, data_luna_np)
