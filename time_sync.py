import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from LUNA.luna_data_to_array import file_to_array


def synchronize_databases(array_ae, array_luna, samples=100, margin_start=0, margin_end=10):

    def remove_outliers_luna():
        pass

    def remove_outliers_ae():
        pass

    def synchronize():
        pass

    def sanity_check():
        pass

    def package_databases():
        pass

    # Getting timestamps from arrays.
    timestamps_luna = array_luna[:, 0] - array_luna[0, 0]
    timestamps_ae = array_ae[:, 0]

    values_ae = array_ae[:, 4]

    # Getting the intervals from LUNA.
    intervals_np = [timestamps_luna[i + 1] - timestamps_luna[i] for i in range(len(timestamps_luna) - 1)]
    intervals_pd = pd.DataFrame(intervals_np, dtype=float)

    intervals_counts = [i[0] for i in intervals_pd.value_counts().index.tolist()]
    intervals_big = np.max(intervals_counts[:2])
    intervals_small = np.min(intervals_counts[:2])
    intervals_mean = np.std(intervals_np)

    # Cutting outliers LUNA end.
    cut_luna_end = -1

    for i in range(len(timestamps_luna)):
        if timestamps_luna[-i] - timestamps_luna[-i - 1] > intervals_mean:
            cut_luna_end = -i

    # Cutting outliers LUNA start.
    cut_luna_start = 0

    for i in range(len(timestamps_luna)):
        if intervals_big + 10 > intervals_np[i] > intervals_big - 10:
            cut_luna_start = i
            break
        elif intervals_small + 10 > intervals_np[i] > intervals_small - 10:
            cut_luna_start = i
            break

    timestamps_luna = timestamps_luna[cut_luna_start: cut_luna_end]

    # Translating LUNA and AE to synchronize start at zero time.
    translation_ae = np.mean(timestamps_ae[:samples])
    translation_luna = timestamps_luna[0] - intervals_big

    timestamps_ae = timestamps_ae - translation_ae
    timestamps_luna = timestamps_luna - translation_luna

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

    # FROM HERE NOT RELEVANT FOR OTHER MAX #

    # Sanity check if data is correctly aligned.
    mean_ae, std_ae = np.mean(values_ae), np.std(values_ae)

    bar = mean_ae - 1 * std_ae

    values_ae = np.array([1 if value_ae > bar else 0 for value_ae in values_ae[cut_ae_start: cut_ae_end]])

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

    return (cut_ae_start, cut_ae_end, translation_ae), (cut_luna_start, cut_luna_end, translation_luna)


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
