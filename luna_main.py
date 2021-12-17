from LUNA.luna_data_to_array import data_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import k_means, mean_shift, aff_prop, agglo, array_to_cluster, cluster_to_image
from LUNA.luna_plotting import plot_arrays, plot_example_cluster
from LUNA.luna_array_to_cluster import print_scores_of_clusters

import os


def demo(panel, file):
    plot_array = False
    plot_k_means = False
    plot_mean_shift = False
    plot_aff_prop = False
    plot_agglo = False

    path = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{file}'

    # load data
    array_left, array_right, labels_left, labels_right = data_to_array(panel, path)

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

    if plot_array:
        # plot all of the images of left and right foot
        plot_arrays(image_left, time_derivative_image_left, length_derivative_image_left,
                    delta_length_left, delta_time_left, panel)

        plot_arrays(image_right, time_derivative_image_right, length_derivative_image_right,
                    delta_length_right, delta_time_right, panel, left=False)

    if plot_k_means:
        time_derivative_array_right_reshaped = time_derivative_array_right.reshape(-1, 1)

        k_means_cluster, k_means_values = k_means(time_derivative_array_right_reshaped)
        k_means_cluster_array = k_means_cluster.reshape(time_derivative_array_right.shape)

        print_scores_of_clusters(time_derivative_array_right_reshaped, k_means_cluster.flatten(),
                                 panel, 'K means')

        plot_example_cluster(time_derivative_image_right, k_means_cluster_array, 'K-means', k_means_values,
                             delta_length_right, delta_time_right)

    if plot_mean_shift:
        time_derivative_array_right_reshaped = time_derivative_array_right.reshape(-1, 1)

        mean_shift_cluster, mean_shift_values = mean_shift(time_derivative_array_right_reshaped)
        mean_shift_cluster_array = mean_shift_cluster.reshape(time_derivative_array_right.shape)

        print_scores_of_clusters(time_derivative_array_right_reshaped, mean_shift_cluster.flatten(),
                                 panel, 'Mean shift')

        plot_example_cluster(time_derivative_image_right, mean_shift_cluster_array, 'Mean shift', mean_shift_values,
                             delta_length_right, delta_time_right)

    if plot_aff_prop:
        time_derivative_array_right_reshaped = time_derivative_array_right.reshape(-1, 1)

        aff_prop_cluster, aff_prop_values = aff_prop(time_derivative_array_right_reshaped)
        aff_prop_cluster_array = aff_prop_cluster.reshape(time_derivative_array_right.shape)

        print_scores_of_clusters(time_derivative_array_right_reshaped, aff_prop_cluster.flatten(),
                                 panel, 'affinity propagation')

        plot_example_cluster(time_derivative_image_right, aff_prop_cluster_array, 'Affinity propagation', aff_prop_values,
                             delta_length_right, delta_time_right)

    if plot_agglo:
        time_derivative_array_right_reshaped = time_derivative_array_right.reshape(-1, 1)

        agglo_cluster, agglo_values = agglo(time_derivative_array_right_reshaped)
        agglo_cluster_array = agglo_cluster.reshape(time_derivative_array_right.shape)

        print_scores_of_clusters(time_derivative_array_right_reshaped, agglo_cluster.flatten(),
                                 panel, 'agglomerative clustering')
        plot_example_cluster(time_derivative_image_right, agglo_cluster_array, 'Agglomerative', agglo_values,
                             delta_length_right, delta_time_right)

    final_cluster = array_to_cluster(time_derivative_array_left, time_derivative_array_right,
                                     length_derivative_array_left, length_derivative_array_right)

    final_cluster_image = cluster_to_image(final_cluster)

    plot_example_cluster(time_derivative_image_right, final_cluster_image, panel, [1, 2], delta_length_right + delta_length_left, delta_time_left)


# USER INPUT #

panel = 'L1-05'
file = 'L1-05-2.txt'

# END USER INPUT #

demo(panel, file)
