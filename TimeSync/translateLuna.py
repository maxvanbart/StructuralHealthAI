import numpy as np


def calc_translation_coeff(timestamps_luna, ribbon_lst):
    """A function which calculates the best time shift based on the luna timestamps and the ribbon list"""
    edge_list = []
    for ribbon in ribbon_lst:
        edge_list.append(ribbon.t_start)
        edge_list.append(ribbon.t_end)

    print("Loaded LUNA data...")
    error_dict = {}
    for dt in range(-200, 200):
        t_luna = np.copy(timestamps_luna)
        t_luna = t_luna + dt
        error_dict[dt] = 0
        # This code is very inefficient
        for i in t_luna:
            # /!\ THIS FUNCTION IS AS INEFFICIENT AS IT CAN BE /!\
            min_dif = None
            for j in edge_list:
                dif = abs(i - j)
                if min_dif is None:
                    min_dif = dif
                elif dif < min_dif:
                    min_dif = dif
            error_dict[dt] += min_dif
        # print(dt, error_dict[dt])
    print("Finished calculating absolute errors...")

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
