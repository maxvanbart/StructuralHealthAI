import numpy as np
import pandas as pd
import scipy.io
import os
import datetime

from PZT.feature_extractor import relative_amp_calc, duration_calc, rise_time_calc, energy_calc, travel_time_calc, \
    calculate_threshold, avgfreq_calc


class TestPZT:
    def __init__(self, name, location, panel):
        # basic object information
        self.name = name
        self.location = f"{location}/{name}"
        self.frequency = int(self.name[:-11])*1000

        # complex actual data
        self.matlab_array = {}
        self.feature_dict = {}

        self.panel = panel

    def load(self):
        # open all the actionneur related files for easy data loading
        for actionneur in os.listdir(self.location):
            # open all the matlab files present in the directory of the actionneur
            y = os.listdir(self.location + '/' + actionneur)
            self.feature_dict[actionneur] = {}

            # Matlab files loaded as dictionary files
            z = [scipy.io.loadmat(f"{self.location}/{actionneur}/{x}") for x in y]
            z = np.array([x['Time_Response'] for x in z])
            z = np.mean(z, axis=0)

            # store the data in a pandas dataframe for easy viewing
            header = ['time', 'chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8']
            self.matlab_array[actionneur] = pd.DataFrame(data=z, columns=header)

    def analyse(self):
        for actionneur in self.matlab_array:
            #######################################################
            # here we collect all the features that we want to use
            #######################################################

            data = np.array(self.matlab_array[actionneur])

            # get the maximum amplitudes
            maximum_column = np.max(data, axis=0)[1:]
            minimum_column = np.min(data, axis=0)[1:]
            abs_column = (abs(minimum_column) + maximum_column)*0.5
            # relative amplitude: amplitude relative to channel 1
            relative_amp_column = relative_amp_calc(maximum_column)
            # Threshold
            threshold = calculate_threshold(maximum_column, self.panel)
            # duration: time from first threshold crossing to last
            duration_column = duration_calc(data, threshold)
            # rise time: time from first threshold crossing to maximum amplitude
            rise_time_column = rise_time_calc(data, threshold)
            # energy: area under the squared signal envelope
            energy_column = energy_calc(data, threshold)
            # travel time: time between first threshold crossing of emitter to first threshold crossing of receiver
            travel_time_column = travel_time_calc(data, threshold)
            # average frequency: the average frequency of the waveform that crosses the threshold.
            avg_freq_column = avgfreq_calc(data, threshold)

            # stack all the features together to compile to pandas dataframe
            index = np.array(range(1, 9))
            z = np.vstack((index, maximum_column, minimum_column, abs_column, relative_amp_column, duration_column,
                           rise_time_column, travel_time_column, energy_column, avg_freq_column))
            z = np.transpose(z)

            # put everything in a dataframe for easy storage
            header = ['index', 'max_amp', 'min_amp', 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time',
                      'travel_time', 'energy', "avg_freq"]
            df = pd.DataFrame(data=z, columns=header)
            self.feature_dict[actionneur] = df

    def __repr__(self):
        return f"TestPZT({self.name})"


class StatePZT:
    def __init__(self, name, location, panel):
        # simple information
        self.name = name
        self.location = location + '/' + name
        self.state_number = int(name.split('_')[1])

        # analysis stuff
        self.frequency_dict = dict()
        self.start_time = convert_to_datetime(self.name[-19:])
        self.panel = panel
        self.test_lst = [TestPZT(x, self.location, self.panel) for x in get_subfolders(self.location)]
        self.f_list = [x.frequency for x in self.test_lst]

    @staticmethod
    def initialize_pzt(name, panel):
        location = f"Files/{name}/PZT/"
        folders = get_subfolders(location)
        subfolder_dict = dict()
        for folder in folders:
            subfolders = get_subfolders(location + folder)
            subfolders = [StatePZT(x, location + folder, panel) for x in subfolders]
            subfolder_dict[folder] = subfolders
        return subfolder_dict

    def analyse(self):
        for test in self.test_lst:
            test.load()
            test.analyse()

        for test in self.test_lst:
            self.frequency_dict[test.frequency] = test.feature_dict

        return self.frequency_dict, self.state_number

    def flatten_db(self):
        dict2 = self.frequency_dict
        if dict2 != {}:
            dict1 = {x: flatten_act(dict2[x]) for x in dict2}
            lst = []
            for f in dict1:
                n = int(f)
                df = dict1[f]
                df['frequency'] = n
                lst.append(df)
            big_df = lst[0]
            lst = lst[1:]
            for item in lst:
                big_df = big_df.append(item, ignore_index=True)
            big_df['time'] = self.start_time
            return big_df
        else:
            return None

    def get_matlab_array(self):
        return self.test_lst

    def __repr__(self):
        return f"StatePZT({self.name})"

    def __lt__(self, other):  # added this for ordering of dictionary keys
        if self.start_time < other.start_time:  # select on date/time
            return True
        return False


def get_subfolders(location):
    entries = os.scandir(location)
    lst = list()
    for entry in entries:
        if entry.is_dir():
            lst.append(entry.name)
    return lst


def convert_to_datetime(time):
    # 2019_01_07_14_02
    year, month, day, hour, minute, second = time.split('_')

    year, month, day = int(year), int(month), int(day)
    hour, minute, second = int(hour), int(minute), int(second)

    date_obj = datetime.datetime(year, month, day, hour, minute, second)
    return datetime.datetime.timestamp(date_obj)


def flatten_act(dictionary):
    lst = []
    for actionneur in dictionary:
        n = int(actionneur.strip('Actionneur'))
        df = dictionary[actionneur]
        df['actionneur'] = n
        lst.append(df)
    big_df = lst[0]
    lst = lst[1:]
    for item in lst:
        big_df = big_df.append(item, ignore_index=True)
    return big_df
