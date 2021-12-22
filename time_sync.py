import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from LUNA.luna_data_to_array import file_to_array


def time_sync(array_ae, array_luna, samples=100, margin_start=0, margin_end=10):
    # Getting timestamps from arrays.
    timestamps_luna = array_luna[:, 0] - array_luna[0, 0]
    timestamps_ae = array_ae[:, 0]

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
    values_luna = np.ones(len(timestamps_luna))
    values_ae = np.zeros(len(timestamps_ae))

    image_array_luna = np.vstack((timestamps_luna, values_luna))
    image_array_ae = np.vstack((timestamps_ae, values_ae))

    sample_y_values_LUNA = np.ones((len(timestamps_luna))) * 0.000035
    sample_y_values_AE = data_ae_np[cut_ae_start: cut_ae_end, 4]

    plt.scatter(timestamps_ae, sample_y_values_AE)
    plt.scatter(timestamps_luna, sample_y_values_LUNA)
    plt.show()

    return (cut_ae_start, cut_ae_end, translation_ae), (cut_luna_start, cut_luna_end, translation_luna)


# opening the files
panel = 'L1-03'
path_luna = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{panel}.txt'
path_ae = os.path.dirname(__file__) + f'/Files/{panel}/AE/{panel}.csv'

# to arrays
data_ae_pd_unsorted = pd.read_csv(path_ae)
data_ae_np_unsorted = data_ae_pd_unsorted.to_numpy(dtype=float)

data_ae_np = data_ae_np_unsorted[np.argsort(data_ae_np_unsorted[:, 0])]
data_luna_np, _, _, _ = file_to_array(panel, path_luna)

time_sync(data_ae_np, data_luna_np)
