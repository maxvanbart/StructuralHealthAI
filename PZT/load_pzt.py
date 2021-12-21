import numpy as np
import pandas as pd
import scipy.io
import os


class TestPZT:
    def __init__(self, name, location):
        self.name = name
        self.location = f"{location}/{name}"
        self.matlab_array = None
        self.max_amp_dict = {}
        self.frequency = int(self.name[:-11])*1000

    def analyse(self):
        for actionneur in os.listdir(self.location):
            y = os.listdir(self.location + '/' + actionneur)
            self.max_amp_dict[actionneur] = {}

            # Matlab files loaded as dictionary files
            z = [scipy.io.loadmat(f"{self.location}/{actionneur}/{x}") for x in y]
            z = [x['Time_Response'] for x in z]
            self.matlab_array = z

            for i in range(1, 9):
                max_lst = list()
                for j in range(len(z)):
                    max_lst.append(max(z[j][:, i]))
                self.max_amp_dict[actionneur][i] = sum(max_lst) / len(max_lst)
        #print(self.max_amp_dict)

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
            test.analyse()

        for test in self.test_lst:
            # ['amplitude']
            self.frequency_dict[test.frequency] = (test.max_amp_dict)
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


"""
entries = os.scandir(files_folder)
        lst = []

        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name, debug=debug, debug_graph=debug_graph))
        return lst
"""
