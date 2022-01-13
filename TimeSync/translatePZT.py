import numpy as np
from matplotlib import pyplot as plt


def calc_translation_coeff(pzt_time, luna_time, graphing=False):
    lst = []
    for i in range(len(luna_time) - 1):
        if 1000 < abs(luna_time[i] - luna_time[i + 1]) < 2000:
            lst.append(abs(luna_time[i] - luna_time[i + 1]))

    # print(f"Average large gap size: {np.average(lst)}")
    dd = np.average(lst) / 2

    error_dict = {}
    # Explore ranges of posible dt values and increase the resolution of smaller ranges as we
    # close in on a value which is precise in whole seconds
    ddt = 1000
    best_dt = 0
    while ddt >= 1:
        error_dict, best_dt = explore_range(pzt_time, luna_time, dd, error_dict, -ddt*100+best_dt, ddt*100+best_dt, ddt)
        ddt = int(ddt/10)

    print(f"Found best dt to be {best_dt} with an error of {error_dict[best_dt]}.")

    # This graph plots the attempted time shifts and their respective errors
    if graphing:
        x = np.transpose(np.array([list(error_dict.keys())]))
        y = np.transpose(np.array([list(error_dict.values())]))
        z = np.hstack((x, y))
        z = z[z[:, 0].argsort()]

        plt.plot(z[:, 0], z[:, 1])
        plt.scatter(z[:, 0], z[:, 1], s=4)
        plt.title(f'Mean error plot')
        plt.xlabel("Time shift dt [s]")
        plt.ylabel("Error [-]")
        plt.show()

    return best_dt, error_dict[best_dt]


def explore_range(pzt_time, luna_time, desired_distance, error_dict, dt_start, dt_end, ddt):
    shift_range = range(dt_start, dt_end, ddt)
    for dt in shift_range:
        if dt not in error_dict.keys():
            error_dict[dt] = shift_error(pzt_time, list(luna_time), dt, desired_distance)

    # Extract the best value for dt from the database
    best_dt = None
    min_error = None
    for i in error_dict:
        if min_error is None:
            min_error = error_dict[i]
            best_dt = i
        elif error_dict[i] < min_error:
            min_error = error_dict[i]
            best_dt = i
    return error_dict, best_dt


def shift_error(pzt_time, luna_time, dt, desired_gap):
    """This function calculates the sum of the errors for a specific time shift"""
    pzt_time = np.array(pzt_time) + dt
    error_sum = 0
    for i in pzt_time:
        error_sum += minimum_dif(i, luna_time, desired_gap)
    average_error = error_sum/len(pzt_time)
    # print(dt, average_error)
    return average_error


def minimum_dif(y, edge_list, z):
    """This function calculates the minimum difference of x and the entries of the edge list"""
    q = [abs(x-y) for x in edge_list]
    return abs(min(q)-z)
