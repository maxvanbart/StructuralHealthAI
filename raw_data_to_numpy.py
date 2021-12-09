import numpy as np
import pandas as pd

import datetime


folder = 'data_LUNA/'
file = 'L1-03.txt'


def raw_to_numpy(folder_path, file_path):

    with open(folder_path + file_path) as file:
        lines = file.readlines()
        feature_labels = lines[0].strip().split('\t')
        feature_labels.append('timestamp')
        data_raw = []

        for line in lines[1:]:
            data_raw.append(line.strip().split('\t'))

        data_raw_array = np.array(data_raw)

        dates = data_raw_array[:, 0]
        timestamps = []

        for date in dates:
            year, month, rest = date.split('-')
            day, rest = rest.split('T')
            hour, minute, second = rest.split(':')

            microseconds = round((float(second) - np.floor(float(second))) * 10 ** 3)

            date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(np.floor(float(second))), int(microseconds))
            timestamp = datetime.datetime.timestamp(date)
            timestamps.append(timestamp)

        feature_time = np.array(timestamps).reshape(-1, 1)
        features_strain = data_raw_array[:, 1:]

        complete_array = np.hstack((features_strain, feature_time))
        complete_dataframe = pd.DataFrame(complete_array, columns=feature_labels)

        print(complete_dataframe.shape)

#
#
# import datetime
#
# x = datetime.datetime(2020, 5, 17)
# y = datetime.datetime.timestamp(x)
#
# print(x)
# print(y)

raw_to_numpy(folder, file)
