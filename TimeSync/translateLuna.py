import numpy as np
from matplotlib import pyplot as plt


def calc_translation_coeff(luna_data, ribbon_lst):
    """A function which calculates the best time shift based on the luna timestamps and the ribbon list"""
    timestamps_luna = np.copy(luna_data)
    edge_list = []
    for ribbon in ribbon_lst:
        edge_list.append(ribbon.t_start)
        edge_list.append(ribbon.t_end)

    print("Loaded LUNA data...")
    error_dict = {}
    shift_range = range(-1000, 1000, 10)
    for dt in shift_range:
        error_dict[dt] = shift_error(np.copy(timestamps_luna), edge_list, dt, penalty='MRE')
    print("Finished calculating absolute errors...")

    # Small plot to show the errors per time
    plt.plot(shift_range, error_dict.values())
    plt.title('Mean error plot')
    plt.xlabel("Time shift dt")
    plt.ylabel("Error")
    plt.show()

    best_dt = None
    min_error = None
    for i in error_dict:
        if min_error is None:
            min_error = error_dict[i]
            best_dt = i
        elif error_dict[i] < min_error:
            min_error = error_dict[i]
            best_dt = i
    print(f"Found best dt to be {best_dt}")
    return best_dt


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

    print(dt, average_error)
    return average_error


def minimum_dif(y, edge_list, penalty='MAE'):
    """This function calculates the minimum difference of x and the entries of the edge list"""
    # /!\ THIS FUNCTION IS AS INEFFICIENT AS IT CAN BE /!\
    if penalty == 'MAE':
        dif_list = [abs(x-y) for x in edge_list]
    elif penalty == 'RMS':
        dif_list = [abs(x - y)**2 for x in edge_list]
    elif penalty == 'MRE':
        dif_list = [abs(x - y) ** 0.5 for x in edge_list]
    # This place could benefit from a custom minimum function as the dif_list should be sorted
    return min(dif_list)
