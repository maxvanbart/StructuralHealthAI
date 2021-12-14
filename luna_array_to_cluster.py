from luna_data_to_array import raw_to_array
from luna_data_to_array import gradient_arrays
from luna_data_to_array import array_to_image
from luna_data_to_array import normalize_array
from luna_data_to_array import gradient_arrays
from luna_data_to_array import plot_images
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cl

#test:
# --- USER INPUT ---
#folder = 'Files/L1-05/LUNA/'
#file = 'L1-05-2.txt'
#panel = 'L1-05'
# ------------------

def get_imputed_arrays(folder, file, panel):
    array_left, array_right, labels_left, labels_right = raw_to_array(folder, file, panel)
    array_left_imp = np.nan_to_num(array_left)
    array_right_imp = np.nan_to_num(labels_right)
    return array_left_imp, array_right_imp, labels_left, labels_right

def test_kmeans(panel):
    array_left, array_right, labels_left, labels_right = raw_to_array(f"Files/{panel[:5]}/LUNA/", f"{panel}.txt", f"{panel[:5]}")
    array_left_time, array_left_length = gradient_arrays(array_left)
    kmeans = cl.KMeans(n_clusters=8, random_state=0).fit(array_left_time)

test_kmeans('L1-05-2')
