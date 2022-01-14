import numpy as np
from matplotlib import pyplot as plt


def calc_translation_coeff(pzt_time, luna_time, pzt_start_point, graphing=False):
    lst = []
    # Fill the list with the gaps between the LUNA points
    for i in range(len(luna_time) - 1):
        # We want to find the larger gaps which tend to be about 1070 in size
        if 1000 < abs(luna_time[i] - luna_time[i + 1]) < 2000:
            lst.append(abs(luna_time[i] - luna_time[i + 1]))

    # We use half the average of the large gaps as the desired distance which a point should be from
    # the closest LUNA point in order to be placed in a correct location
    dd = np.average(lst) / 2

    error_dict = {}
    # Explore ranges of posible dt values and increase the resolution of smaller ranges as we
    # close in on a value which is precise in whole seconds
    ddt = 1000
    best_dt = 0
    while ddt >= 1:
        # Fill the error dict in the specified range using the specified resolution and
        # center the range on the previous best value for dt
        error_dict, best_dt = explore_range(pzt_time, luna_time, pzt_start_point, dd, error_dict, -ddt*100+best_dt, ddt*100+best_dt, ddt)
        # Increase the resolution
        ddt = int(ddt/10)

    print(f"Found best dt for the PZT time shift to be {best_dt} with an error of {error_dict[best_dt]}.")
    # This graph plots the attempted time shifts and their respective errors
    if graphing:
        # Create an array to store the values used for the plotting
        x = np.transpose(np.array([list(error_dict.keys())]))
        y = np.transpose(np.array([list(error_dict.values())]))
        z = np.hstack((x, y))
        z = z[z[:, 0].argsort()]

        # Plot the error graph
        plt.plot(z[:, 0], z[:, 1])
        plt.scatter(z[:, 0], z[:, 1], s=4)
        plt.title(f'Mean error plot')
        plt.xlabel("Time shift dt [s]")
        plt.ylabel("Error [-]")
        plt.show()
    return best_dt, error_dict[best_dt]


def explore_range(pzt_time, luna_time, pzt_start_point, desired_distance, error_dict, dt_start, dt_end, ddt):
    """This function explores a specified range of timeshift values and adds them to the error dictionary"""
    # Turn the range parameters into a range
    shift_range = range(dt_start, dt_end, ddt)
    # Iterate through all the values in the shift range and try adding them to the error dictionary
    for dt in shift_range:
        # If the range is already present in the error dictionary we will
        # skip the calculations as they have already been done
        if dt not in error_dict.keys():
            error_dict[dt] = shift_error(pzt_time, list(luna_time), dt, desired_distance)
            error_dict[dt] += shift_error_front(pzt_start_point, luna_time, dt)

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
    # We calculate the error by summing the individual errors
    for i in pzt_time:
        error_sum += minimum_dif_2(i, luna_time, desired_gap)
    average_error = error_sum/len(pzt_time)
    # print(dt, average_error)
    return average_error


def shift_error_front(pzt_start_point, luna_time, dt):
    """ This function calculates the shift errors for the most forward points as they should always be to
        the left of the rest of the data """
    pzt_start_point = np.array(pzt_start_point) + dt
    final_error = 0
    # This function uses a different metric for error as these points should be placed different than the others
    for i in pzt_start_point:
        q = [i-x for x in luna_time]
        # print(q)
        if max(q) > 0:
            final_error += max(q)
    # print(dt, final_error)
    return 10*final_error


def minimum_dif_2(y, edge_list, z):
    """This function calculates the minimum difference of x and the entries of the edge list"""
    # This function brute forces once again however that is not a problem since the amount of values is way lower
    q = [abs(x-y) for x in edge_list]
    return abs(min(q)-z)
