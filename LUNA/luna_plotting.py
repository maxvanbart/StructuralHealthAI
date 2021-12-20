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


def plot_cluster(im_time_left, im_time_right, im_length_left, im_length_right, im_cluster_left, im_cluster_right,
                 length_left, length_right, time, panel):

    figure = plt.figure(constrained_layout=True)
    figure.supxlabel('Length measurements [-]')
    figure.supylabel('Time measurements [-]')
    figure.suptitle(f'Panel {panel}')

    sub_figures = figure.subfigures(1, 3)
    sub_figures[0].suptitle('Time derivatives')
    sub_figures[1].suptitle('Length derivatives')
    sub_figures[2].suptitle('Clusters')

    axs0 = sub_figures[0].subplots(1, 2, sharey=True)
    axs0[0].imshow(im_time_left, extent=[0, length_left, 0, time], aspect='auto')
    axs0[0].set_title('Left')

    axs0[1].imshow(im_time_right, extent=[0, length_right, 0, time], aspect='auto')
    axs0[1].set_title('Right')

    axs1 = sub_figures[1].subplots(1, 2, sharey=True)
    axs1[0].imshow(im_length_left, extent=[0, length_left, 0, time], aspect='auto')
    axs1[0].set_title('Left')

    axs1[1].imshow(im_length_right, extent=[0, length_right, 0, time], aspect='auto')
    axs1[1].set_title('Right')

    axs2 = sub_figures[2].subplots(1, 2, sharey=True)
    axs2[0].imshow(im_cluster_left, extent=[0, length_left, 0, time], aspect='auto', cmap='gray')
    axs2[0].set_title('Left')

    axs2[1].imshow(im_cluster_right, extent=[0, length_right, 0, time], aspect='auto', cmap='gray')
    axs2[1].set_title('Right')

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
