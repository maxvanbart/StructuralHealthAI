import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from LUNA.luna_data_to_array import file_to_array

# opening the files
panel = 'L1-09'
file_luna = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{panel}.txt'
file_ae = os.path.dirname(__file__) + f'/Files/{panel}/AE/{panel}.csv'

# to arrays
data_ae_pd_unsorted = pd.read_csv(file_ae)
data_ae_np_unsorted = data_ae_pd_unsorted.to_numpy(dtype=float)

data_ae_np = data_ae_np_unsorted[np.argsort(data_ae_np_unsorted[:, 0])]
data_luna_np, _, _, _ = file_to_array(panel, file_luna)

# setting LUNA start time to 0
timestamps_LUNA = data_luna_np[:, 0] - data_luna_np[0, 0]
timestamps_AE = data_ae_np[:, 0]

# cutting outliers LUNA
cut = -1

intervals = [timestamps_LUNA[i + 1] - timestamps_LUNA[i] for i in range(len(timestamps_LUNA) - 1)]

intervals_pd = pd.DataFrame(intervals, dtype=float)

interval_counts = intervals_pd.value_counts().index.tolist()

print(interval_counts)

main_values = [interval_counts[0][0], interval_counts[1][0]]

mean_interval = np.std(intervals)

# cutting end
for i in range(len(timestamps_LUNA)):
    if timestamps_LUNA[-i] - timestamps_LUNA[-i - 1] > mean_interval:
        cut = -i

timestamps_LUNA = timestamps_LUNA[:cut]

cut = 0

# cutting start
for i in range(len(timestamps_LUNA)):
    if main_values[0] + 10 > intervals[i] > main_values[0] - 10:
        cut = i
        break
    elif main_values[1] + 10 > intervals[i] > main_values[1] - 10:
        cut = i
        break

timestamps_LUNA = timestamps_LUNA[cut:]

y_values = np.ones((len(timestamps_LUNA))) * 0.000035

translation_ae = np.mean(timestamps_AE[:100])

timestamps_AE = timestamps_AE - translation_ae

translation_luna = timestamps_LUNA[0] - 252

timestamps_LUNA = timestamps_LUNA - translation_luna

cut_ae_start = 0
cut_ae_end = timestamps_LUNA[-1] + 252 + 10

timestamps_AE = [i for i in timestamps_AE if 0 <= i <= cut_ae_end]

# cutting outliers AE


plt.scatter(timestamps_AE, data_ae_np[:len(timestamps_AE), 4])
plt.scatter(timestamps_LUNA, y_values)
plt.show()
