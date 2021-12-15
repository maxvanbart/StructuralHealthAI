import matplotlib.colors as mpl
import matplotlib.pyplot as plt

import numpy as np

from LUNA.luna_data_to_array import raw_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import k_means, mean_shift, aff_prop, agglo


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


def plot_cluster(image_time, cluster_vector, cluster_name, cluster_values, length, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Reference')

    image_cluster = np.flip(cluster_vector, axis=0)

    plt.subplot(1, 2, 2)
    plt.imshow(image_cluster, extent=[0, length, 0, time], cmap='inferno')
    plt.xlabel(f'Number of clusters: {len(cluster_values)}')
    plt.ylabel('Timestamp [-]')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{cluster_name} clustering')

    plt.show()


def demo(panel):

    # load data
    array_left, array_right, labels_left, labels_right = raw_to_array(panel)

    # get indices for begin left and right
    delta_length_left = float(labels_left[-1]) - float(labels_left[0])
    delta_length_right = float(labels_right[-1]) - float(labels_right[0])

    delta_time_left = len(array_left)
    delta_time_right = len(array_right)

    # get the two derivatives of the two arrays
    time_derivative_array_left, length_derivative_array_left = gradient_arrays(array_left[:, 1:])
    time_derivative_array_right, length_derivative_array_right = gradient_arrays(array_right[:, 1:])

    # plot image of left and right foot
    image_left = array_to_image(array_left)
    image_right = array_to_image(array_right)

    # plot time and space derivative image of left foot
    time_derivative_image_left = array_to_image(time_derivative_array_left)
    length_derivative_image_left = array_to_image(length_derivative_array_left)

    # plot time and space derivative image of right foot
    time_derivative_image_right = array_to_image(time_derivative_array_right)
    length_derivative_image_right = array_to_image(length_derivative_array_right)

    # plot all of the images of left and right foot
    plot_arrays(image_left, time_derivative_image_left, length_derivative_image_left,
                delta_length_left, delta_time_left, panel)

    plot_arrays(image_right, time_derivative_image_right, length_derivative_image_right,
                delta_length_right, delta_time_right, panel, left=False)

    # k-means
    k_means_cluster, k_means_values = k_means(panel)

    plot_cluster(time_derivative_image_right, k_means_cluster, 'K-means', k_means_values,
                 delta_length_right, delta_time_right)

    # mean shift
    mean_shift_cluster, mean_shift_values = mean_shift(panel)

    plot_cluster(time_derivative_image_right, mean_shift_cluster, 'Mean shift', mean_shift_values,
                 delta_length_right, delta_time_right)

    # affinity propagation

    aff_prop_cluster, aff_prop_values = aff_prop(panel)

    plot_cluster(time_derivative_image_right, aff_prop_cluster, 'Affinity propagation', aff_prop_values,
                 delta_length_right, delta_time_right)

    # agglomerative clustering

    agglo_cluster, agglo_values = agglo(panel)

    plot_cluster(time_derivative_image_right, agglo_cluster, 'Agglomerative clustering', agglo_values,
                 delta_length_right, delta_time_right)


demo('L1-03')
