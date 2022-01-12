import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math


def calc_translation_coeffs(luna_data, ribbon_lst, luna_vector, final_time, graphing=False):
    """A function which calculates the best time shift based on the luna timestamps and the ribbon list"""
    timestamps_luna = np.copy(luna_data)
    # Here we collect a list of all the timestamps for the edges of the ribbons
    edge_list = []
    for ribbon in ribbon_lst:
        edge_list.append(ribbon.t_start)
        edge_list.append(ribbon.t_end)

    # Here we make a dataframe from the luna timestamps and the luna vector in order to seperate by source
    timestamps_luna = np.transpose(np.array([timestamps_luna]))
    df = pd.DataFrame(data=np.hstack((timestamps_luna, luna_vector)), columns=["timestamp", "source"])

    # This block generates a list of arrays for each unique index
    indexes = df['source'].unique().tolist()
    grouped = df.groupby(df.source)
    groups = {}
    for index in indexes:
        groups[index] = np.array(grouped.get_group(index))[:, 0]

    print("Loaded LUNA data...")

    best_dts = {}
    error_dicts = {}
    best_errors = {}
    for group in groups:
        group_data = groups[group]
        error_dict = {}
        # t0 = group_data[0]

        error_dict, best_dt, best_error = explore(error_dict, group_data, edge_list, final_time)
        print("Finished calculating absolute errors...")
        error_dicts[group] = error_dict
        best_dts[group] = best_dt
        best_errors[group] = best_error

        # Small plot to show the errors per time
        if graphing:
            x = np.transpose(np.array([list(error_dict.keys())]))
            y = np.transpose(np.array([list(error_dict.values())]))
            z = np.hstack((x, y))
            z = z[z[:, 0].argsort()]

            plt.plot(z[:, 0], z[:, 1])
            plt.scatter(z[:, 0], z[:, 1], s=4)
            plt.title(f'Mean error plot for group {int(group)}')
            plt.xlabel("Time shift dt [s]")
            plt.ylabel("Error [-]")
            plt.show()
    return best_dts, best_errors


def explore(error_dict, group_data, edge_list, final_time):
    magnitude = int(math.log(final_time, 10)-2)
    ddt = int(10**magnitude)
    best_dt = 0
    # We loop the explore_range function a number of times in order to refine the areas of interest
    while ddt >= 1:
        error_dict, best_dt = explore_range(error_dict, group_data, edge_list, -ddt*100+best_dt, ddt*100+best_dt, ddt)
        ddt = int(ddt/10)
    # error_dict, best_dt = explore_range(error_dict, group_data, edge_list, -10000, 10000, 100)
    # # First we explore a large region to get a rough estimate of the best value for dt
    # error_dict, best_dt = explore_range(error_dict, group_data, edge_list, -1000+best_dt, 1000+best_dt, 10)
    # # Explore around the best dt to refine the result
    # error_dict, best_dt = explore_range(error_dict, group_data, edge_list, -10+best_dt, 10+best_dt, 1)

    print(f"Found best dt to be {best_dt} with an error of {error_dict[best_dt]}.")
    return error_dict, best_dt, error_dict[best_dt]


def explore_range(error_dict, group_data, edge_list, dt_start, dt_end, ddt):
    shift_range = range(dt_start, dt_end, ddt)
    for dt in shift_range:
        if dt not in error_dict.keys():
            error_dict[dt] = shift_error(np.copy(group_data), edge_list, dt, penalty='MAE')

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


def shift_error(t_luna, edge_list, dt, penalty='MAE'):
    """This function calculates the sum of the errors for a specific time shift"""
    if penalty not in ['RMS', 'MAE', 'MRE']:
        raise ValueError
    t_luna = t_luna + dt
    error_sum = 0
    for i in t_luna:
        error_sum += minimum_dif(i, edge_list, penalty=penalty)
    average_error = error_sum/len(t_luna)
    if penalty == 'RMS':
        average_error = average_error**0.5
    elif penalty == 'MRE':
        average_error = average_error**2

    # print(dt, average_error)
    return average_error


def minimum_dif(y, edge_list, penalty='MAE'):
    """This function calculates the minimum difference of x and the entries of the edge list"""
    min_dif = abs(edge_list[0] - y)
    x = 1
    while x < len(edge_list) and abs(edge_list[x] - y) < min_dif:
        min_dif = abs(edge_list[x] - y)
        x += 1

    if penalty == 'MAE':
        min_dif = min_dif
    elif penalty == 'RMS':
        min_dif = min_dif**2
    elif penalty == 'MRE':
        min_dif = min_dif**0.5
    else:
        raise ValueError
    # This place could benefit from a custom minimum function as the dif_list should be sorted
    return min_dif
