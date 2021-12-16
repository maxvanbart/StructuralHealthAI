from LUNA.luna_data_to_array import raw_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import k_means, mean_shift, aff_prop, agglo
from LUNA.luna_plotting import plot_arrays, plot_cluster
from LUNA.luna_array_to_cluster import print_scores_of_clusters
import os


def demo(panel, file):

    path = os.path.dirname(__file__) + f'/Files/{panel}/LUNA/{file}'

    # load data
    array_left, array_right, labels_left, labels_right = raw_to_array(panel, path)

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

    if False:
        # plot all of the images of left and right foot
        plot_arrays(image_left, time_derivative_image_left, length_derivative_image_left,
                    delta_length_left, delta_time_left, panel)

        plot_arrays(image_right, time_derivative_image_right, length_derivative_image_right,
                    delta_length_right, delta_time_right, panel, left=False)


    if True:
        k_means_cluster, k_means_values = k_means(time_derivative_array_right)
        print_scores_of_clusters(time_derivative_array_right, k_means_cluster.ravel(), panel, "K means")
        plot_cluster(time_derivative_image_right, k_means_cluster, 'K_means', k_means_values,
                    delta_length_right, delta_time_right)

    # mean shift
    if True:
        mean_shift_cluster, mean_shift_values = mean_shift(time_derivative_array_right)
        print_scores_of_clusters(time_derivative_array_right, mean_shift_cluster.ravel(), panel, "Mean shift")
        plot_cluster(time_derivative_image_right, mean_shift_cluster, 'Mean shift', mean_shift_values,
                     delta_length_right, delta_time_right)

    # affinity propagation
    if True:
        aff_prop_cluster, aff_prop_values = aff_prop(time_derivative_array_right)
        print_scores_of_clusters(time_derivative_array_right, aff_prop_cluster.ravel(), panel, "affinity propagation")
        plot_cluster(time_derivative_image_right, aff_prop_cluster, 'Affinity propagation', aff_prop_values,
                     delta_length_right, delta_time_right)

    # agglomerative clustering
    if True:
        agglo_cluster, agglo_values = agglo(time_derivative_array_right)
        print_scores_of_clusters(time_derivative_array_right, agglo_cluster.ravel(), panel, "agglomerative clustering")
        plot_cluster(time_derivative_image_right, agglo_cluster, 'Agglomerative', agglo_values,
                     delta_length_right, delta_time_right)


# USER INPUT #

panel = 'L1-03'
file = f'{panel}.txt'

# END USER INPUT #

demo(panel, file)
