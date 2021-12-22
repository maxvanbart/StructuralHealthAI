import numpy as np
import pandas as pd

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


def duration_calc(data, threshold):
    """duration: time from first threshold crossing to last"""
    pass


def rise_time_calc(data, threshold):
    """rise time: time from first threshold crossing to maximum amplitude"""
    pass


def energy_calc(data, threshold):
    """energy: area under the squared signal envelope"""
    pass


def travel_time_calc(data, threshold):
    """travel time: time between first threshold crossing of emitter to first threshold crossing of receiver"""
    pass
