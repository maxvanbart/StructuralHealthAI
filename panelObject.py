import os

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster

from LUNA.luna_data_to_array import raw_to_array, gradient_arrays

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, debug=False):
        self.name = name
        self.ae_database = None
        self.luna_database = None
        self.debug = debug

    @staticmethod
    def initialize_all():
        """A static method which checks the folders present and generates a Panel object for every folder"""
        entries = os.scandir(files_folder)
        lst = []

        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name))

        return lst

    # All the AE related code for the object
    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv()
        # print(self.ae_database.hits)
        print(f"Successfully loaded AE data for {self.name}.")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        self.ae_clustered_database = init_clustering(self.ae_database, debug=self.debug)
        # self.ae_database.corr_matrix()
        freq_amp_cluster(self.ae_clustered_database)
        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        # LUNA code related to loading stuff should go here

        luna_data_left, luna_data_right, labels_left, labels_right = raw_to_array(self.name)
        luna_data_left_time, luna_data_left_length = gradient_arrays(luna_data_left)
        luna_data_right_time, luna_data_right_length = gradient_arrays(luna_data_right)

        self.luna_database = [luna_data_left_time, luna_data_right_time, luna_data_left_length, luna_data_right_length]

        print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        # LUNA code relating to analysis should go here
        print(f"Successfully analysed LUNA data for {self.name}.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
