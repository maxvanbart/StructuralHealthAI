import numpy as np
import os
from LUNA.luna_data_to_array import folder_to_array, file_to_array
from LUNA.luna_array_to_cluster import agglo, mean_shift, aff_prop
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering


def cluster_luna(timestamps):

    # clustered_timestamps, types = agglo(timestamps.reshape(-1, 1), scaled=True, n=None)

    clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=3000).fit(timestamps.reshape(-1, 1))
    clustered_timestamps = clustering.labels_

    values = np.ones(len(timestamps))

    print(clustered_timestamps)
    print(values.shape)
    print(timestamps.shape)

    plt.scatter(timestamps, values, c=clustered_timestamps)
    plt.show()


# opening the files
panel = 'L1-23'
path_luna = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{panel}-2.txt'
path_ae = os.path.dirname(__file__) + f'/Files/{panel}/AE/{panel}-clustered.csv'

folder_path = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/'

data_luna_np, _ = folder_to_array(panel, folder_path)
# data_luna_np, _, _, _ = file_to_array(panel, path_luna)

timestamps_luna = data_luna_np[:, 0]

cluster_luna(timestamps_luna)
