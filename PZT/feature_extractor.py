import numpy as np
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


def calculate_threshold(max_amp, panel):
    return max_amp * panel.pzt_threshold


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
    time_array = data[:, 0]
    data = data[:, 1:]

    # Extract first positive threshold crossing
    data_list = list(data)
    data_list = [list(x) for x in data_list]
    new_data = np.array([[1 if x >= threshold[y.index(x)] else 0 for x in y] for y in data_list])
    start_index = list(np.argmax(new_data, axis=0))
    first_threshold_crossing_time = np.array([time_array[x] for x in start_index])

    # Extract time for maximum threshold
    max_amp_crossing_index = data.argmax(axis=0)
    max_amp_time = np.array([time_array[x] for x in max_amp_crossing_index])
    return relatify(max_amp_time - first_threshold_crossing_time)


def energy_calc(data, threshold):
    """energy: area under the squared signal envelope"""
    time_array = data[:, 0]
    data = data[:, 1:]
    dt = time_array[1]  # seconds
    energy_riemann = abs(data)*dt
    energy_column = np.sum(energy_riemann, axis=0)
    return relatify(energy_column)


def travel_time_calc(data, threshold):
    """travel time: time between first threshold crossing of emitter to first threshold crossing of receiver"""
    time_array = data[:, 0]
    data = data[:, 1:]

    # Extract first positive threshold crossing
    data_list = list(data)
    data_list = [list(x) for x in data_list]
    new_data = np.array([[1 if x >= threshold[y.index(x)] else 0 for x in y] for y in data_list])
    start_index = list(np.argmax(new_data, axis=0))
    first_threshold_crossing_time = np.array([time_array[x] for x in start_index])
    travel_time = first_threshold_crossing_time - first_threshold_crossing_time[0]
    return travel_time


def avgfreq_calc(data, threshold, debugging=False):
    """returns average frequency of waveform that crosses the origin"""
    duration_array = duration_calc(data, threshold)
    time_array = data[:, 0]
    data = data[:, 1:]
    count_array = np.zeros(duration_array.shape)
    for row_ndx in range(data.shape[1]):
        counts, prev = 0, None
        for current in data[:, row_ndx]:
            if prev is not None:
                if prev < 0 < current or prev > 0 > current:
                    counts += 1
            prev = current

        if debugging:
            plt.plot(time_array, data[:, row_ndx])
            # plt.plot(time_array, np.full(len(data[:, row_ndx]), threshold[row_ndx]))
            plt.show()

        count_array[row_ndx] = counts

    avgfreq_array = np.zeros(duration_array.shape)

    for ndx in range(avgfreq_array.shape[0]):
        avgfreq_array[ndx] = count_array[ndx] / time_array[-1]

    return avgfreq_array


def relative_freq_calc(data, threshold):
    return


def relatify(column):
    c0 = column[0]
    column = column/c0
    # print(c0)
    return column
