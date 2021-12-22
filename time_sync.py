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
data_ae_pd = pd.read_csv(file_ae)
data_ae_np = data_ae_pd.to_numpy(dtype=float)
data_luna_np, _, _, _ = file_to_array(panel, file_luna)

# setting LUNA start time to 0
timestamps_LUNA = data_luna_np[:, 0] - data_luna_np[0, 0]

cut = 0

# cutting outliers LUNA
mean_interval = np.std([timestamps_LUNA[i + 1] - timestamps_LUNA[i] for i in range(len(timestamps_LUNA) - 1)])

for i in range(len(timestamps_LUNA)):
    if timestamps_LUNA[-i] - timestamps_LUNA[-i - 1] > mean_interval:
        cut = -i

timestamps_LUNA = timestamps_LUNA[:cut]
y_values = np.ones((len(timestamps_LUNA))) * 0.000035

# ------ works up until here --------

cut_ae = -1

timestamps_AE = data_ae_np[:cut_ae, 0]

mean_dev = np.std([timestamps_AE[i + 1] - timestamps_AE[i] for i in range(len(timestamps_AE) - 1)])

for i in range(len(timestamps_AE) - 1):
    if timestamps_AE[i + 1] - timestamps_AE[i] > mean_dev:
        correct_time = timestamps_AE[i]
        break

interval = timestamps_LUNA[0] - correct_time

timestamps_LUNA = timestamps_LUNA - 110

plt.scatter(timestamps_AE, data_ae_np[:cut_ae, 4])
plt.scatter(timestamps_LUNA, y_values)
plt.show()
