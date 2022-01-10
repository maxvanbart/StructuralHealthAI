import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TimeSync.translateLuna import calc_translation_coeff
from TimeSync.dataTypeClasses import Ribbon


def sync_time(ae_df, array_luna, name='Generic Panel', bin_width=1):
    # Convert the AE dataframe to a numpy array
    array = np.array(ae_df)

    # Determine the highest value of t and use it to create the bins which will be used
    final_time = ae_df[ae_df['time'] == ae_df['time'].max()]
    final_time = int(np.ceil(np.array(final_time['time'])[0]) + 1)
    print("Final value of t:", final_time)

    bin_count = int(np.ceil(final_time / bin_width))
    bins = [0] * bin_count
    trange = range(0, bin_count)

    for row in list(array):
        # x is the time (t) at which this point occurs
        x = row[0]
        # y is the index of the bin in which this point should be sorted
        y = int(x // bin_width)
        bins[y] += 1
    print("Finished sorting to bins...")

    # Here we sort all the bins into ribbons in order to sort them together
    ribbon_lst = []
    found_prev = False
    not_found_count = 99
    for i in trange:
        # If the current bin has any datapoints it will be added to a ribbon
        if bins[i] > 0:
            not_found_count = 0
            if found_prev:
                # If there was a previous non-empty bin found we will add it to that ribbon
                ribbon_lst[-1].add_entry(i, bins[i])
            else:
                # Otherwise, we start a new ribbon
                ribbon_lst.append(Ribbon(bin_width))
                ribbon_lst[-1].add_entry(i, bins[i])
                found_prev = True
        else:
            # We tolerate having one bin with no values before we consider a ribbon finished
            # /!\ THE CURRENT VALUE OF ZERO SHOULD STILL BE TUNED /!\
            if not_found_count > 0:
                found_prev = False
            else:
                not_found_count += 1

    # Here we trigger all ribons to update their defining properties using the bins that they are associated with
    for ribbon in ribbon_lst:
        ribbon.update()
    print(f"Generated {len(ribbon_lst)} ribbons...")

    # Here we sort out all ribbons with a width less than 3 as these widths are never associated with actual ribbons
    ribbon_lst = [x for x in ribbon_lst if x.width > 3]

    # Here we itteratively decrease the standard deviation of the ribbon widths in order to sort out any other
    # abnormaly thin ribbons
    ribbon_width_lst = [x.width for x in ribbon_lst]
    while np.std(ribbon_width_lst) > 15:
        mean_width = np.mean(ribbon_width_lst)
        ribbon_lst = [x for x in ribbon_lst if x.width > mean_width]
        ribbon_width_lst = [x.width for x in ribbon_lst]

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
    sample_y_values_LUNA = np.ones((len(timestamps_luna)))
    timestamps_luna_shifted = np.copy(timestamps_luna) + best_dt
    plt.scatter(timestamps_luna_shifted, sample_y_values_LUNA, c='red', s=4)
    # plt.scatter(timestamps_luna, sample_y_values_LUNA, c='green', s=4)
    plt.title(name)
    plt.xlabel("Time [s]")
    plt.ylabel("404")
    plt.show()
