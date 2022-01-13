import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import energy_time_cluster, freq_amp_cluster
from AE.feature_extraction import frequency_extraction
from AE.clustering import clustering_time_energy

from PZT.analyze_pzt import analyse_pzt
from PZT.load_pzt import StatePZT

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import array_to_cluster
from LUNA.luna_plotting import plot_cluster, plot_clusters
from LUNA.luna_preprocessing import preprocess_array
from LUNA.luna_postprocessing import filter_array

from TimeSync.timeSync import sync_time

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, debug=False, debug_graph=False):
        # General
        self.name = name
        self.debug = debug
        self.debug_graph = debug_graph

        # AE
        self.ae_start_time = None
        self.ae_database = None
        self.ae_clustered_database = None

        # LUNA
        self.luna_database = None
        self.luna_database_derivatives = None
        self.luna_database_clustered = None
        self.luna_database_filtered = None

        self.luna_file_vector = None

        self.luna_time_labels = None
        self.luna_length_labels = None

        self.luna_time_shift_errors = None
        self.luna_time_shift_vector = None

        self.folder_parent = os.path.dirname(__file__)
        self.folder_ae = None
        self.folder_luna = self.folder_parent + f'/Files/{self.name}/LUNA/'

        # PZT
        self.pzt_database = None
        self.pzt_clustered_database = None
        self.pzt_start_times = None

    @staticmethod
    def initialize_all(debug=False, debug_graph=False):
        """A static method which checks the folders present and generates a Panel object for every folder"""
        entries = os.scandir(files_folder)
        lst = []

        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name, debug=debug, debug_graph=debug_graph))
        return lst

    # All the AE related code for the object
    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv()
        # print(self.ae_database.hits)
        print(f"Successfully loaded AE data for {self.name}.")

    def analyse_ae(self, force_clustering=False):
        """Function to analyse the AE data in the folder"""
        # Try to find a clustered file else cluster the data
        location = 'Files/' + self.name + "/AE/" + self.name + "-clustered.csv"
        try:
            if force_clustering:
                raise FileNotFoundError
            self.ae_clustered_database = pd.read_csv(location)
            print(f"Successfully loaded clustered AE data for {self.name}.")
        except FileNotFoundError:
            print('Clustered file not found, clustering data...')
            # creation of clustered database
            self.ae_clustered_database = self.ae_database.hits
            # detection of energy outliers
            self.ae_clustered_database["energy_outlier"] = energy_time_cluster(self.ae_clustered_database)
            # removal of the energy outlier
            self.ae_clustered_database = self.ae_clustered_database[self.ae_clustered_database["energy_outlier"] == 1]
            # hits combination
            self.ae_clustered_database = init_clustering(self.ae_clustered_database, debug=self.debug)
            # adding frequency to the database
            self.ae_clustered_database["frequency"] = frequency_extraction(self.ae_clustered_database)
            # frequency outlier detection
            self.ae_clustered_database["frequency_outlier"] = freq_amp_cluster(self.ae_clustered_database)
            # adding extracted features and clusters
            print(f"Clustering completed for {self.name}, features and clusters being added to database...")
            # create new CSV
            pd.DataFrame(self.ae_clustered_database).to_csv(location, index=False)
        self.ae_clustered_database = self.ae_clustered_database.sort_values(by=['time'])

        # self.ae_database.corr_matrix()
        # energy_time_cluster(self.ae_clustered_database, plotting=True)
        # batch_fre_amp_clst(self.ae_clustered_database)

        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        luna_data_left, luna_data_right, self.luna_file_vector, labels_left, labels_right = \
            folder_to_array(self.name, self.folder_luna)
        luna_data_left = preprocess_array(np.hstack((luna_data_left, self.luna_file_vector)))
        luna_data_right = preprocess_array(np.hstack((luna_data_right, self.luna_file_vector)))

        self.luna_file_vector = luna_data_left[:, -1]

        self.luna_database = [luna_data_left[:, 1: -1], luna_data_right[:, 1: -1]]

        self.luna_time_labels = luna_data_left[:, 0]
        self.luna_length_labels = [labels_left, labels_right]

        print(f"Successfully loaded LUNA data for {self.name}...")

    def synchronise_luna(self):
        """Function which takes all the internal variables related to the seperate sensors and time synchronises them"""
        sv, e = sync_time(self.ae_database.hits, self.luna_database[0], self.luna_file_vector, name=self.name)
        self.luna_time_shift_vector = sv
        self.luna_time_shift_errors = e
        self.luna_time_labels = self.luna_time_labels + self.luna_time_shift_vector

        print(self.luna_time_shift_errors)
        print(f"Successfully synchronized time for {self.name}...")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""

        # 1. get time and length derivatives.
        left_time, left_length = gradient_arrays(self.luna_database[0])
        right_time, right_length = gradient_arrays(self.luna_database[1])

        # 2. get clustered database.
        self.luna_database_derivatives = [left_time, right_time, left_length, right_length]
        self.luna_database_clustered = array_to_cluster(left_time, right_time, left_length, right_length)

        # 3. filter original database with clustered database.
        left_filtered = filter_array(self.luna_database[0], self.luna_database_clustered[0], self.luna_time_labels, self.luna_length_labels[0])
        right_filtered = filter_array(self.luna_database[1], self.luna_database_clustered[1], self.luna_time_labels, self.luna_length_labels[1])

        self.luna_database_filtered = [left_filtered, right_filtered]

        print(f"Successfully analysed LUNA data for {self.name}...")

    def visualize_luna(self):
        image_left_time = array_to_image(self.luna_database_derivatives[0])
        image_right_time = array_to_image(self.luna_database_derivatives[1])

        image_left_length = array_to_image(self.luna_database_derivatives[2])
        image_right_length = array_to_image(self.luna_database_derivatives[3])

        image_left = (image_left_time + image_left_length) / 2
        image_right = (image_right_time + image_right_length) / 2

        plot_clusters(image_left, image_right, len(self.luna_length_labels[0]), len(self.luna_length_labels[1]),
                      len(self.luna_time_labels), self.luna_time_labels, self.name)

    def load_pzt(self):
        self.pzt_database = StatePZT.initialize_pzt(self.name)
        time_list = []
        for identifier in self.pzt_database:
            time_list += [x.start_time for x in self.pzt_database[identifier]]
        time_list.sort()
        self.pzt_start_times = time_list
        print(f"Successfully loaded PZT data for {self.name}.")

    def analyse_pzt(self, force_clustering=False):
        location = 'Files/' + self.name + "/PZT/" + self.name + "_PZT-clustered.csv"

        try:
            if force_clustering:
                raise FileNotFoundError
            print(f"Successfully loaded clustered PZT data for {self.name}.")
            self.pzt_clustered_database = pd.read_csv(location)
        except FileNotFoundError:
            print('Clustered PZT file not found, clustering data...')
            # The part where all the data is analyzed
            analyse_pzt(self.pzt_database, self.name)
            # The part where all the data boils up
            lst = []
            for folder in self.pzt_database:
                for state in self.pzt_database[folder]:
                    lst.append(state.flatten_db())

            # find the first dataframe in the list
            final_i = None
            big_df = None

            for i in range(len(lst)):
                if lst[i] is not None:
                    big_df = lst[i]
                    final_i = i
                    break

            if big_df is None:
                raise ValueError

            # delete the dataframe from the list as to prevent a copy from showing up
            if final_i is not None:
                del lst[final_i]
            else:
                raise ValueError

            for item in lst:
                if item is not None:
                    big_df = big_df.append(item, ignore_index=True)

            self.pzt_clustered_database = big_df
            pd.DataFrame(self.pzt_clustered_database).to_csv(location, index=False)
            print("Successfully created PZT clustered .csv.")

        # call plotting function
        print(f"Successfully analysed PZT data for {self.name}.")

    def time_synchronise(self):
        """Function which takes all the internal variables related to the separate sensors and time synchronises them"""
        pass

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
