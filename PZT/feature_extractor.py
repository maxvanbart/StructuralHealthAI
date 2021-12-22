import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# all functions should return a 1x8 numpy array


def relative_amp_calc(maximum_column):
    """relative amplitude: max amplitude relative to channel 1"""
    relative_max_column = np.copy(maximum_column)
    max_value = relative_max_column[0]
    for numb, item in enumerate(relative_max_column):
        relative_max_column[numb] = item / max_value
    if relative_max_column[0] != 1:
        print("mistake in relative_amp_cal")
        pass
    return relative_max_column


def calculate_threshold(max_amp):
    return max_amp * 0.1


def duration_calc(database, threshold, debug_graph=False):
    """duration: time from first threshold crossing to last"""
    # Extract the time column from the provided database
    time = database[:, 0]
    data = list(database[:, 1:])
    # Turn all the internal numpy arrays into lists for list comprehension
    data = [list(x) for x in data]
    threshold = list(threshold)

    # We replace all values which are above the threshold with 1, otherwise we replace them with 0
    data = np.array([[1 if x >= threshold[y.index(x)] else 0 for x in y] for y in data])

    # Debug graph to see if the calculated data alligns with the graphs
    if debug_graph:
        for i in range(8):
            plt.scatter(time, data[:, i])
            plt.plot(database[:, 0], database[:, i+1])
        plt.show()

    # Here we use the argmax to find the index of the first occurance of a value above the threshold
    start_index = list(np.argmax(data, axis=0))
    data_flipped = np.flip(data, axis=0)
    end_index_flipped = list(np.argmax(data_flipped, axis=0))
    end_index = list(np.array([data.shape[0]-1]*8) - end_index_flipped)

    # We turn the indices of the start and end time into the actual time and calculate the duration
    start_time = np.array([time[x] for x in start_index])
    end_time = np.array([time[x] for x in end_index])
    return end_time - start_time


def rise_time_calc(data, threshold):
    """rise time: time from first threshold crossing to maximum amplitude"""
    pass


def energy_calc(data, threshold):
    """energy: area under the squared signal envelope"""
    pass


def travel_time_calc(data, threshold):
    """travel time: time between first threshold crossing of emitter to first threshold crossing of receiver"""
    pass
