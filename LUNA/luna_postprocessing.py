import numpy as np


def filter_array(array, array_cluster, array_row_labels, array_column_labels):
    array_filtered = []
    rows, columns = array.shape

    for row in range(rows):
        for column in range(columns):
            if array_cluster[row, column]:
                if not np.isnan(array[row, column]):
                    array_filtered.append([float(array_row_labels[row]), float(array_column_labels[column]), array[row, column]])

    return np.array(array_filtered)
