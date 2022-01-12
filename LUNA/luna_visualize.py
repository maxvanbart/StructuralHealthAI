def visualize_luna(luna_database_derivatives, luna_database_clustered, name):
    time_left, time_right, length_left, length_right = luna_database_derivatives
    cluster_left, cluster_right = luna_database_clustered

    image_time_left = array_to_image(time_left)
    image_time_right = array_to_image(time_right)

    image_length_left = array_to_image(length_left)
    image_length_right = array_to_image(length_right)

    image_cluster_left = cluster_to_image(cluster_left)
    image_cluster_right = cluster_to_image(cluster_right)

    time, delta_length_left = time_left.shape
    time, delta_length_right = time_right.shape

    plot_cluster(image_time_left, image_time_right, image_length_left, image_length_right,
                 image_cluster_left, image_cluster_right, delta_length_left, delta_length_right, time, name)
