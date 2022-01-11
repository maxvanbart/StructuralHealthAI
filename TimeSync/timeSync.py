import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TimeSync.translateLuna import calc_translation_coeff
from TimeSync.dataTypeClasses import Ribbon
from TimeSync.ribbonFinder import sort_ribbons, purge_ribbons


def sync_time(ae_df, array_luna, vector_luna_source, name='Generic Panel', bin_width=1):
    print(vector_luna_source)

    # Convert the AE dataframe to a numpy array
    array = np.array(ae_df)

    # Determine the highest value of t and use it to create the bins which will be used
    final_time = ae_df[ae_df['time'] == ae_df['time'].max()]
    final_time = int(np.ceil(np.array(final_time['time'])[0]) + 1)
    print("Final value of t:", final_time)

    # Initialize the bins for the bin based method
    bin_count = int(np.ceil(final_time / bin_width))
    bins = [0] * bin_count

    for row in list(array):
        # x is the time (t) at which this point occurs
        x = row[0]
        # y is the index of the bin in which this point should be sorted
        y = int(x // bin_width)
        bins[y] += 1
    print("Finished sorting to bins...")

    # This function is used to combine all the bins into data ribbons
    trange = range(0, bin_count)
    ribbon_lst = sort_ribbons(bins, trange, bin_width)

    # remove all ribbons which are abnormally small
    ribbon_lst = purge_ribbons(ribbon_lst)

    # LUNA stuff, all previous AE stuff should still be migrated to its own module
    timestamps_luna = array_luna[:, 0] - array_luna[0, 0]
    best_dt = calc_translation_coeff(timestamps_luna, ribbon_lst)

    # Final Plot
    # This plot shows how accurate the results are
    # Plot ribbons as horizontal lines
    for ribbon in ribbon_lst:
        # print(str(ribbon))
        # print(ribbon.t_start, ribbon.t_end)
        plt.plot([ribbon.t_start, ribbon.t_end], [1, 1], 'b')
    # Plot LUNA points as red dots
    sample_y_values_luna = np.ones((len(timestamps_luna)))
    timestamps_luna_shifted = np.copy(timestamps_luna) + best_dt
    plt.scatter(timestamps_luna_shifted, sample_y_values_luna, c='red', s=4)
    # plt.scatter(timestamps_luna, sample_y_values_LUNA, c='green', s=4)
    plt.title(name)
    plt.xlabel("Time [s]")
    plt.ylabel("404")
    plt.show()
