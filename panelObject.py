import os

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster

from LUNA.luna_data_to_array import file_to_array, gradient_arrays
from LUNA.luna_array_to_cluster import k_means

files_folder = "Files"


class Panel:
    """An object which represents a array"""
    def __init__(self, name, debug=False):
        self.name = name

        self.ae_database = None
        self.ae_database_clustered = None

        self.luna_database = None
        self.luna_database_clustered = None

        self.debug = debug

        self.folder_parent = os.path.dirname(__file__)
        self.folder_ae = None
        self.folder_luna = self.folder_parent + f'/Files/{self.name}/LUNA/'

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
        self.ae_database_clustered = init_clustering(self.ae_database, debug=self.debug)
        # self.ae_database.corr_matrix()
        freq_amp_cluster(self.ae_database_clustered)
        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        path = self.folder_luna + f'{self.name}.txt'

        luna_data_left, luna_data_right, labels_left, labels_right = file_to_array(self.name, path)
        luna_data_left_time, luna_data_left_length = gradient_arrays(luna_data_left)
        luna_data_right_time, luna_data_right_length = gradient_arrays(luna_data_right)

        self.luna_database = [luna_data_left_time, luna_data_right_time, luna_data_left_length, luna_data_right_length]

        print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""

        # EXAMPLE CLUSTER, THIS HAS TO CHANGE #

        self.luna_database_clustered = k_means(self.luna_database[1])

        print(f"Successfully analysed LUNA data for {self.name}.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
