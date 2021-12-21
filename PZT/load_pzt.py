import numpy as np
import pandas as pd
import scipy.io
import os
import datetime


class TestPZT:
    def __init__(self, name, location):
        # basic object information
        self.name = name
        self.location = f"{location}/{name}"
        self.frequency = int(self.name[:-11])*1000

        # complex actual data
        self.matlab_array = {}
        self.feature_dict = {}

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
            # here we collect all the features that we want to use
            data = np.array(self.matlab_array[actionneur])
            maximum_column = np.max(data, axis=0)[1:]
            minimum_column = np.min(data, axis=0)[1:]
            abs_column = (abs(minimum_column) + maximum_column)*0.5
            # relative amplitude: amplitude relative to channel 1
            # duration: time from first threshold crossing to last
            # rise time: time from first threshold crossing to maximum amplitude
            # energy: area under the squared signal envelope
            # travel time: time between first threshold crossing of emitter to first threshold crossing of receiver

            # stack all the features together to compile to pandas dataframe
            index = np.array(range(1, 9))
            z = np.vstack((index, maximum_column, minimum_column, abs_column))
            z = np.transpose(z)

            # put everything in a dataframe for easy storage
            header = ['index', 'max_amp', 'min_amp', 'avg_abs_amp']
            df = pd.DataFrame(data=z, columns=header)
            print(df)
            self.feature_dict[actionneur] = df

    def __repr__(self):
        return f"TestPZT({self.name})"


class StatePZT:
    def __init__(self, name, location):
        # simple information
        self.name = name
        self.location = location + '/' + name
        self.test_lst = [TestPZT(x, self.location) for x in get_subfolders(self.location)]
        self.state_number = int(name.split('_')[1])
        self.f_list = [x.frequency for x in self.test_lst]

        # analysis stuff
        self.frequency_dict = dict()
        self.start_time = convert_to_datetime(self.name[-19:])


    @staticmethod
    def initialize_pzt(name):
        location = f"Files/{name}/PZT/"
        folders = get_subfolders(location)
        subfolder_dict = dict()
        for folder in folders:
            # print(folder[-19:])
            subfolders = get_subfolders(location + folder)
            subfolders = [StatePZT(x, location + folder) for x in subfolders]
            subfolder_dict[folder] = subfolders
        return subfolder_dict

    def analyse(self):
        # print(self.name, self.state_number)
        for test in self.test_lst:
            test.load()
            test.analyse()

        for test in self.test_lst:
            # ['amplitude']
            self.frequency_dict[test.frequency] = test.max_amp_dict
        return self.frequency_dict, self.state_number

    def get_matlab_array(self):
        return self.test_lst

    def __repr__(self):
        return f"StatePZT({self.name})"


def get_subfolders(location):
    entries = os.scandir(location)
    lst = list()
    for entry in entries:
        if entry.is_dir():
            lst.append(entry.name)
    return lst


def convert_to_datetime(time):
    year, month, day, hour, minute, second = time.split('_')

    year, month, day = int(year), int(month), int(day)
    hour, minute, second = int(hour), int(minute), int(second)

    date_obj = datetime.datetime(year, month, day, hour, minute, second)
    return datetime.datetime.timestamp(date_obj)
