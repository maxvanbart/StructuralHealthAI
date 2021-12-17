import matplotlib.colors as mpl
import matplotlib.pyplot as plt

import numpy as np


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


def plot_cluster(image_time, image_length, image_cluster, panel_name, cluster_name, length, time):
    fig, axs = plt.subplots(1, 3, sharex='all')

    axs[0].imshow(image_time, extent=[0, length, 0, time])
    axs[0].set(xlabel='L [mm]', ylabel='Measurement [-]', title='Time derivative')

    axs[1].imshow(image_length, extent=[0, length, 0, time])
    axs[1].set(xlabel='L [mm]', ylabel='Measurement [-]', title='Length derivative')

    axs[2].imshow(image_cluster, extent=[0, length, 0, time], cmap='gray')
    axs[2].set(xlabel='L [mm]', ylabel='Measurement [-]', title='Cluster')

    plt.suptitle(f'{panel_name} with {cluster_name} clustering')
    plt.show()


def plot_example_cluster(image_time, cluster_array, cluster_name, cluster_values, length, time):
    plt.subplot(1, 2, 1)
    plt.imshow(image_time, extent=[0, length, 0, time])
    plt.xlabel('L [mm]')
    plt.ylabel('Timestamp [-]')
    plt.title('Reference')

    image_cluster = np.flip(cluster_array, axis=0)

    plt.subplot(1, 2, 2)
    plt.imshow(image_cluster, extent=[0, length, 0, time], cmap='inferno')
    plt.xlabel(f'Number of clusters: {len(cluster_values)}')
    plt.ylabel('Timestamp [-]')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{cluster_name} clustering')

    plt.show()
