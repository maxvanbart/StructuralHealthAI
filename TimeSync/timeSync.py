import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TimeSync.translateLuna import calc_translation_coeffs
from TimeSync.dataTypeClasses import Ribbon
from TimeSync.ribbonFinder import sort_ribbons, purge_ribbons


def sync_time(ae_df, array_luna, vector_luna_source, name='Generic Panel', bin_width=1):
    # Convert the AE dataframe to a numpy array
    array = np.array(ae_df)

    # Convert luna vector to vector
    vector_luna_source = np.transpose(np.array([vector_luna_source]))

    # Determine the highest value of t and use it to create the bins which will be used
    final_time = ae_df[ae_df['time'] == ae_df['time'].max()]
    final_time = int(np.ceil(np.array(final_time['time'])[0]) + 1)
    print(f"Final value of t: {final_time}...")

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

    ##############
    # LUNA stuff #
    ##############
    timestamps_luna = array_luna[:, 0] - array_luna[0, 0]
    best_dts, best_errors = calc_translation_coeffs(timestamps_luna, ribbon_lst, vector_luna_source, final_time)

    vector_groups = [x[0] for x in list(np.copy(vector_luna_source))]
    vector_shifts = np.array([best_dts[x] for x in vector_groups])
    # vector_groups = [int(x) for x in vector_groups]

    timestamps_luna_shifted = (np.copy(timestamps_luna) + vector_shifts)

    ##############
    # Final Plot #
    ##############
    # This plot shows how accurate the results are
    # Plot ribbons as horizontal lines
    height_value = -10

    for ribbon in ribbon_lst:
        # print(str(ribbon))
        # print(ribbon.t_start, ribbon.t_end)
        plt.plot([ribbon.t_start, ribbon.t_end], [height_value, height_value], 'b')
    # Plot LUNA points as red dots
    sample_y_values_luna = height_value * np.ones((len(timestamps_luna)))

    plt.scatter(timestamps_luna_shifted, sample_y_values_luna, c='r', s=4)

    plt.plot(range(len(bins)), bins, c='g')
    # plt.scatter(timestamps_luna_shifted, sample_y_values_luna, c=vector_groups, s=4)
    # for i, label in enumerate(vector_groups):
    #     plt.annotate(vector_groups, (timestamps_luna_shifted[i], sample_y_values_luna[i]))
    # plt.scatter(timestamps_luna, sample_y_values_LUNA, c='green', s=4)
    plt.title(name)
    plt.xlabel("Time [s]")
    plt.ylabel("404")
    plt.show()

    return vector_shifts, best_errors
