import matplotlib.colors as mpl
import matplotlib.pyplot as plt

from LUNA.luna_data_to_array import raw_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import k_means, mean_shift


def plot_arrays(image, image_time, image_length, length, time, left=True):
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
        plt.suptitle('Left foot panel L1-03')
    else:
        plt.suptitle('Right foot panel L1-03')

    cbr = plt.colorbar(plt.cm.ScalarMappable(cmap='bwr', norm=mpl.Normalize(vmin=-1, vmax=1)))
    cbr.set_label('Scaled values [-]')
    plt.show()


def plot_cluster(image_time, cluster_vector, cluster_values, cluster_name, length, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Reference')

    plt.subplot(1, 2, 2)
    plt.pcolormesh(cluster_vector, cmap='inferno')
    plt.xlabel(f'Number of clusters: {len(cluster_values)}')
    plt.ylabel('Timestamp [-]')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{cluster_name} clustering')

    plt.show()


def demo(panel):

    # # --- USER INPUT ---
    # panel = 'L1-03'
    # # ------------------

    array_left, array_right, labels_left, labels_right = raw_to_array(panel)

    delta_length_left = float(labels_left[-1]) - float(labels_left[0])
    delta_length_right = float(labels_right[-1]) - float(labels_right[0])

    delta_time_left = len(array_left)
    delta_time_right = len(array_right)

    time_derivative_array_left, length_derivative_array_left = gradient_arrays(array_left[:, 1:])
    time_derivative_array_right, length_derivative_array_right = gradient_arrays(array_right[:, 1:])

    image_left = array_to_image(array_left)
    image_right = array_to_image(array_right)

    time_derivative_image_left = array_to_image(time_derivative_array_left)
    length_derivative_image_left = array_to_image(length_derivative_array_left)

    time_derivative_image_right = array_to_image(time_derivative_array_right)
    length_derivative_image_right = array_to_image(length_derivative_array_right)

    plot_arrays(image_left, time_derivative_image_left, length_derivative_image_left,
                delta_length_left, delta_time_left)

    plot_arrays(image_right, time_derivative_image_right, length_derivative_image_right,
                delta_length_right, delta_time_right, left=False)

    k_means_cluster, k_means_cluster_values = k_means(panel)

    plot_cluster(time_derivative_image_right, k_means_cluster, k_means_cluster_values, 'K-means',
                 delta_length_right, delta_time_right)

    mean_shift_cluster, mean_shift_cluster_values = mean_shift(panel)

    plot_cluster(time_derivative_image_right, mean_shift_cluster, mean_shift_cluster_values, 'Mean shift',
                 delta_length_right, delta_time_right)


