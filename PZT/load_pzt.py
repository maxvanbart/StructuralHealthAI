import numpy as np
import pandas as pd
import scipy.io
import os


class TestPZT:
    def __init__(self, name, location):
        self.name = name
        self.location = f"{location}/{name}"

    def __repr__(self):
        return f"TestPZT({self.name})"


class StatePZT:
    def __init__(self, name, location):
        self.name = name
        self.location = location + '/' + name
        self.test_lst = [TestPZT(x, self.location) for x in get_subfolders(self.location)]

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
