import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TimeSync.translateLuna import calc_translation_coeffs
from TimeSync.dataTypeClasses import Ribbon
from TimeSync.ribbonFinder import sort_ribbons, purge_ribbons
from TimeSync.translatePZT import calc_translation_coeff


def sync_luna(ae_df, vector_luna_source, timestamps_luna, name='Generic Panel', bin_width=1, graphing=False):
    # Convert the AE dataframe to a numpy array
    array = np.array(ae_df)
    # Convert luna vector to vector
    vector_luna_source = np.transpose(np.array([vector_luna_source]))

    # Determine the highest value of t and use it to create the bins which will be used
    final_time = ae_df[ae_df['time'] == ae_df['time'].max()]
    final_time = int(np.ceil(np.array(final_time['time'])[0]) + 1)

    # Initialize the bins for the bin based method.
    bin_count = int(np.ceil(final_time / bin_width))
    bins = [0] * bin_count

    # Fill the bins for the bin based method
    for row in list(array):
        # x is the time (t) at which this point occurs
        x = row[0]
        # y is the index of the bin in which this point should be sorted
        y = int(x // bin_width)
        bins[y] += 1

    # This function is used to combine all the bins into data ribbons
    trange = range(0, bin_count)
    ribbon_lst = sort_ribbons(bins, trange, bin_width)
    # remove all ribbons which are abnormally small
    ribbon_lst = purge_ribbons(ribbon_lst)

    ##############
    # LUNA stuff #
    ##############
    best_dts, best_errors = calc_translation_coeffs(timestamps_luna, ribbon_lst, vector_luna_source, final_time)

    vector_groups = [x[0] for x in list(np.copy(vector_luna_source))]
    vector_shifts = np.array([best_dts[x] for x in vector_groups])
    # vector_groups = [int(x) for x in vector_groups]

    timestamps_luna_shifted = (np.copy(timestamps_luna) + vector_shifts)

    ##############
    # Final Plot #
    ##############
    # This plot shows how accurate the results are
    if graphing:
        height_value = -10

        # Plot ribbons as horizontal lines
        for ribbon in ribbon_lst:
            plt.plot([ribbon.t_start, ribbon.t_end], [height_value, height_value], 'b')

        # Plot LUNA points as red dots
        sample_y_values_luna = height_value * np.ones((len(timestamps_luna)))
        plt.scatter(timestamps_luna_shifted, sample_y_values_luna, c='r', s=4)

        plt.plot(range(len(bins)), bins, c='g')
        plt.title(name)
        plt.xlabel("Time [s]")
        plt.ylabel("404")
        plt.show()
    return vector_shifts, best_errors, ribbon_lst


def sync_pzt(pzt_time, luna_time, ae_ribbons, pzt_file_count, results, name='Generic Panel', graphing=False):
    pzt_time = np.array(pzt_time) - pzt_time[0]

    # There always seem to be n nodes which should be to the left of the other data
    pzt_start_points = pzt_time[:pzt_file_count]
    pzt_time = pzt_time[pzt_file_count:]

    best_dt, best_error = calc_translation_coeff(pzt_time, luna_time, pzt_start_points)

    ####################
    # DATA EXPLORATION #
    # ###################
    # pzt_time = pzt_time + best_dt
    # pzt_start_points = pzt_start_points + best_dt
    # plt.title(f"Time sync plot for panel {name}")
    # plt.xlabel("Time [s]")
    # i = 0
    # for ribbon in ae_ribbons:
    #     plt.plot([ribbon.t_start, ribbon.t_end], [0, 0], 'b', label="AE ribbons" if i == 0 else "")
    #     i += 1
    #
    # plt.scatter(pzt_start_points, [0] * len(pzt_start_points), c='g', s=4, label="PZT measurements")
    # plt.scatter(pzt_time, [0] * len(pzt_time), c='g', s=4, label="")
    # plt.scatter(luna_time, [0] * len(luna_time), c='r', s=4, label="LUNA measurements")
    # plt.legend()
    #
    # plt.savefig(f'{results}/TimeSync_{name}.png')
    # if graphing:
    #     plt.show()

    return best_dt, best_error
