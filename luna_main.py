from LUNA.luna_data_to_array import raw_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import k_means, mean_shift, aff_prop, agglo
from LUNA.luna_plotting import plot_arrays, plot_cluster


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

    # # k-means
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

    plot_cluster(time_derivative_image_right, agglo_cluster, 'Agglomerative', agglo_values,
                 delta_length_right, delta_time_right)


demo('L1-03')
