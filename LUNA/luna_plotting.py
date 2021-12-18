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


# def plot_cluster(im_time_left, im_time_right, im_length_left, im_length_right, im_cluster_left, im_cluster_right,
#                  panel_name, cluster_name, length_left, length_right, time):
#     fig, axs = plt.subplots(1, 6, sharey=True)
#
#     axs[0].imshow(im_time_left, extent=[0, length_left, 0, time])
#     axs[0].set(xlabel='Length measurement [-]', ylabel='Time measurement [-]', title='d/dt left')
#
#     axs[1].imshow(im_time_right, extent=[0, length_right, 0, time])
#     axs[1].set(title='d/dt right')
#     axs[1].sharey(axs[0])
#
#     axs[2].imshow(im_length_left, extent=[0, length_left, 0, time])
#     axs[2].set(xlabel='Length measurement [-]', ylabel='Time measurement [-]', title='d/dx left')
#
#     axs[3].imshow(im_length_right, extent=[0, length_right, 0, time])
#     axs[3].set(xlabel='L [mm]', ylabel='Measurement [-]', title='d/dx right')
#
#     axs[4].imshow(im_cluster_left, extent=[0, length_left, 0, time], cmap='gray')
#     axs[4].set(xlabel='Length measurement [-]', ylabel='Time measurement [-]', title='Cluster left')
#
#     axs[5].imshow(im_cluster_right, extent=[0, length_right, 0, time], cmap='gray')
#     axs[5].set(xlabel='L [mm]', ylabel='Measurement [-]', title='Cluster right')
#
#     plt.suptitle(figure'{panel_name} with {cluster_name} clustering')
#     plt.show()

def option_2():

    figure = plt.figure(constrained_layout=True)
    figure.supxlabel('Length measurements [-]')
    figure.supylabel('Time measurements [-]')
    figure.suptitle('Panel')

    sub_figures = figure.subfigures(1, 3)
    sub_figures[0].suptitle('Time derivatives')
    sub_figures[1].suptitle('Length derivatives')
    sub_figures[2].suptitle('Clusters')

    axs0 = sub_figures[0].subplots(1, 2, sharey=True)
    axs0[0].plot(x, y)
    axs0[0].set_title('Left')

    axs0[1].plot(x, abs(y))
    axs0[1].set_title('Right')
    axs0[1].tick_params(labelleft=False)

    axs1 = sub_figures[1].subplots(1, 2, sharey=True)
    axs1[0].plot(x, y)
    axs1[0].set_title('Left')

    axs1[1].plot(x, abs(y))
    axs1[1].set_title('Right')
    axs1[1].tick_params(labelleft=False)

    axs2 = sub_figures[2].subplots(1, 2, sharey=True)
    axs2[0].plot(x, y)
    axs2[0].set_title('Left')

    axs2[1].plot(x, abs(y))
    axs2[1].set_title('Right')
    axs2[1].tick_params(labelleft=False)

    plt.show()


def plot_cluster(im_time_left, im_length_left, im_cluster_left, panel_name, cluster_name, length_left, time):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0.1)
    ax1, ax2, ax3 = gs.subplots(sharex='col', sharey='row')

    ax1.imshow(im_time_left, extent=[0, length_left, 0, time])
    ax1.set(ylabel='Time measurement [-]', title='d/dt left')

    ax2.imshow(im_length_left, extent=[0, length_left, 0, time])
    ax2.set(xlabel='Length measurement [-]', title='d/dx left')

    ax3.imshow(im_cluster_left, extent=[0, length_left, 0, time], cmap='gray')
    ax3.set(title='Cluster left')

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
