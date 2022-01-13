import numpy as np

from TimeSync.dataTypeClasses import Ribbon


def sort_ribbons(bins, trange, bin_width, max_gap=0):
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
            if not_found_count > max_gap:
                found_prev = False
            else:
                not_found_count += 1

    # Here we trigger all ribons to update their defining properties using the bins that they are associated with
    for ribbon in ribbon_lst:
        ribbon.update()
    print(f"Generated {len(ribbon_lst)} ribbons...")

    return ribbon_lst


def purge_ribbons(ribbon_lst):
    l0 = len(ribbon_lst)
    # Here we sort out all ribbons with a width less than 3 as these widths are never associated with actual ribbons
    ribbon_lst = [x for x in ribbon_lst if x.width > 3]

    # Here we itteratively decrease the standard deviation of the ribbon widths in order to sort out any other
    # abnormaly thin ribbons
    ribbon_width_lst = [x.width for x in ribbon_lst]
    while np.std(ribbon_width_lst) > 15:
        mean_width = np.mean(ribbon_width_lst)
        ribbon_lst = [x for x in ribbon_lst if x.width > mean_width]
        ribbon_width_lst = [x.width for x in ribbon_lst]
    l1 = len(ribbon_lst)
    print(f"Purged {round(100*(abs(l0-l1)/l0), 3)}% of ribbons...")
    return ribbon_lst
