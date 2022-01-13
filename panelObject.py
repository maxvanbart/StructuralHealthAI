import os

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster, all_features_cluster, create_cluster_batches, energy_time_cluster, freq_amp_energy_plot
from AE.clustering import clustering_time_energy
from AE.feature_analysis import freq_amp_cluster, energy_time_cluster
from AE.feature_extraction import frequency_extraction
from PZT.analyze_pzt import analyse_pzt
from PZT.load_pzt import StatePZT

import pandas as pd

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import array_to_cluster, cluster_to_image
from LUNA.luna_plotting import plot_cluster

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
        self.luna_database_clustered = None

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
            print('Clustered AE file not found, clustering data...')
            self.ae_clustered_database = init_clustering(self.ae_database, debug=self.debug)

            # adding extracted features and clusters
            print(f"AE clustering completed for {self.name}, features and clusters being added to database...")
            self.ae_clustered_database["frequency"] = frequency_extraction(self.ae_clustered_database)
            self.ae_clustered_database["energy_outlier"] = energy_time_cluster(self.ae_clustered_database)
            self.ae_clustered_database["frequency_outlier"] = freq_amp_cluster(self.ae_clustered_database)

            # create new CSV
            pd.DataFrame(self.ae_clustered_database).to_csv(location, index=False)

            print("Successfully created AE clustered .csv.")

        # self.ae_database.corr_matrix()
        # freq_amp_cluster(self.ae_clustered_database)
        # all_features_cluster(self.ae_clustered_database)
        # freq_amp_time_cluster(self.ae_clustered_database)
        # energy_time_cluster(self.ae_clustered_database)
        freq_amp_energy_plot(self.ae_database.hits, title="Frequency, amplitude and energy for uncombined randomly sampled emissions in the L1-03 panel")

        # freq_amp_cluster(self.ae_clustered_database, plotting=True)
        # energy_time_cluster(self.ae_clustered_database, plotting=True)
        # batch_fre_amp_clst(self.ae_clustered_database)

        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        luna_data_left, luna_data_right = folder_to_array(self.name, self.folder_luna)

        # First entry each row removed as this is the timestamp!
        luna_data_left_time, luna_data_left_length = gradient_arrays(luna_data_left[:, 1:])
        luna_data_right_time, luna_data_right_length = gradient_arrays(luna_data_right[:, 1:])

        self.luna_database = [luna_data_left_time, luna_data_right_time, luna_data_left_length, luna_data_right_length]

        print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        time_left, time_right, length_left, length_right = self.luna_database

        self.luna_database_clustered = array_to_cluster(time_left, time_right, length_left, length_right)

        print(f"Successfully analysed LUNA data for {self.name}.")

    def plot_luna(self):
        """Plots the final result for LUNA"""
        time_left, time_right, length_left, length_right = self.luna_database
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
            for i in range(len(lst)):
                if lst[i] is not None:
                    big_df = lst[i]
                    final_i = i
                    break

            # delete the dataframe from the list as to prevent a copy from showing up
            del lst[final_i]

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
