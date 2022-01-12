import os
import pandas as pd
import numpy as np

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster, all_features_cluster, create_cluster_batches, energy_time_cluster, freq_amp_energy_plot
from AE.clustering import clustering_time_energy

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import array_to_cluster, cluster_to_image
from LUNA.luna_plotting import plot_cluster

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

        self.luna_file_vector = None
        self.luna_shift_errors = None

        self.folder_parent = os.path.dirname(__file__)
        self.folder_ae = None
        self.folder_luna = self.folder_parent + f'/Files/{self.name}/LUNA/'

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
            print(f"Successfully loaded clustered data for {self.name}.")
        except FileNotFoundError:
            print('Clustered file not found, clustering data...')
            self.ae_clustered_database = init_clustering(self.ae_database, debug=self.debug)
            pd.DataFrame(self.ae_clustered_database).to_csv(location, index=False)
        self.ae_clustered_database = self.ae_clustered_database.sort_values(by=['time'])

        # self.ae_database.corr_matrix()
        # freq_amp_cluster(self.ae_clustered_database)
        # all_features_cluster(self.ae_clustered_database)
        # freq_amp_time_cluster(self.ae_clustered_database)
        # energy_time_cluster(self.ae_clustered_database)
        # freq_amp_energy_plot(self.ae_database.hits, title="Frequency, amplitude and energy for uncombined randomly sampled emissions in the L1-03 panel")

        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        luna_data_left, luna_data_right, self.luna_file_vector = folder_to_array(self.name, self.folder_luna)
        self.luna_database = [luna_data_left, luna_data_right]

        print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        luna_data_left_time, luna_data_left_length = gradient_arrays(self.luna_database[0])
        luna_data_right_time, luna_data_right_length = gradient_arrays(self.luna_database[1])

        self.luna_database_derivatives = [luna_data_left_time, luna_data_right_time,
                                          luna_data_left_length, luna_data_right_length]

        time_left, time_right, length_left, length_right = self.luna_database_derivatives

        self.luna_database_clustered = array_to_cluster(time_left, time_right, length_left, length_right)

        print(f"Successfully analysed LUNA data for {self.name}.")

    def plot_luna(self):
        """Plots the final result for LUNA"""
        time_left, time_right, length_left, length_right = self.luna_database_derivatives
        cluster_left, cluster_right = self.luna_database_clustered

        image_time_left = array_to_image(time_left)
        image_time_right = array_to_image(time_right)

        image_length_left = array_to_image(length_left)
        image_length_right = array_to_image(length_right)

        image_cluster_left = cluster_to_image(cluster_left)
        image_cluster_right = cluster_to_image(cluster_right)

        time, delta_length_left = time_left.shape
        time, delta_length_right = time_right.shape

        plot_cluster(image_time_left, image_time_right, image_length_left, image_length_right,
                     image_cluster_left, image_cluster_right, delta_length_left, delta_length_right, time, self.name)

    def time_synchronise(self):
        """Function which takes all the internal variables related to the seperate sensors and time synchronises them"""
        sv, e = sync_time(self.ae_clustered_database, self.luna_database[0], self.luna_file_vector, name=self.name)
        shift_vector = sv

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
