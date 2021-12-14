from luna_data_to_array import raw_to_array
from luna_data_to_array import gradient_arrays
from luna_data_to_array import array_to_image
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

def kmeans(panel):
    #get array
    array_left, array_right, labels_left, labels_right = raw_to_array(f"Files/{panel[:5]}/LUNA/", f"{panel}.txt", f"{panel[:5]}")
    array_right_time, array_right_length = gradient_arrays(array_right)

    #clustering
    kmeans = cl.KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(array_right_time)

    #output
    return clusters.reshape(-1,1)

def mean_shift(panel):
    #get array
    array_left, array_right, labels_left, labels_right = raw_to_array(f"Files/{panel[:5]}/LUNA/", f"{panel}.txt", f"{panel[:5]}")
    array_right_time, array_right_length = gradient_arrays(array_right)

    #clustering
    mean_shift = cl.MeanShift()
    clusters = mean_shift.fit_predict(array_right_time)

    #output
    return clusters.reshape(-1,1)

#testing:
print(kmeans('L1-05-2'))
print(mean_shift('L1-05-2'))