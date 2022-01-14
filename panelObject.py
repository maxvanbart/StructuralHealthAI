import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import energy_time_cluster, freq_amp_cluster
from AE.feature_extraction import frequency_extraction

from PZT.analyze_pzt import analyse_pzt, make_clusters
from PZT.load_pzt import StatePZT

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays
from LUNA.luna_array_to_cluster import array_to_cluster
from LUNA.luna_preprocessing import preprocess_array
from LUNA.luna_postprocessing import filter_array

from TimeSync.timeSync import sync_luna, sync_pzt

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, debug=False, debug_graph=False, force_clustering=False):
        # General
        self.name = name
        self.debug = debug
        self.debug_graph = debug_graph
        self.force_clustering = force_clustering

        # AE
        self.ae_database = None
        self.ae_clustered_database = None
        self.ae_ribbons = None

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
        self.folder_luna = self.folder_parent + f'/Files/{self.name}/LUNA/'

        # PZT
        self.pzt_database = None
        self.pzt_clustered_database = None
        self.pzt_start_times = None
        self.pzt_dt = None

    @staticmethod
    def initialize_all(debug=False, debug_graph=False, force_clustering=False):
        """A static method which checks the folders present and generates a Panel object for every folder"""
        if force_clustering:
            print("Force clustering is set to True, all datafiles will be regenerated...")
        entries = os.scandir(files_folder)
        lst = []

        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name, debug=debug, debug_graph=debug_graph, force_clustering=force_clustering))
        return lst

    # All the AE related code for the object
    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv()
        # print(self.ae_database.hits)
        print(f"Successfully loaded AE data for {self.name}...")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        # Try to find a clustered file else cluster the data
        location = 'Files/' + self.name + "/AE/" + self.name + "-clustered.csv"
        try:
            if self.force_clustering:
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

        self.luna_time_labels = luna_data_left[:, 0] - luna_data_left[0, 0]
        self.luna_length_labels = [labels_left, labels_right]

        print(f"Successfully loaded LUNA data for {self.name}...")

    def synchronise_luna(self):
        """Function which takes all the internal variables related to the seperate sensors and time synchronises them"""
        sv, e, rb = sync_luna(self.ae_database.hits, self.luna_file_vector, self.luna_time_labels, name=self.name)
        self.luna_time_shift_vector = sv
        self.luna_time_shift_errors = e

        self.luna_time_labels = self.luna_time_labels + self.luna_time_shift_vector
        self.ae_ribbons = rb

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

    # All PZT related code for the object
    def load_pzt(self):
        self.pzt_database = StatePZT.initialize_pzt(self.name)
        time_list = []
        for identifier in self.pzt_database:
            time_list += [x.start_time for x in self.pzt_database[identifier]]
        time_list.sort()
        self.pzt_start_times = time_list
        print(f"Successfully loaded PZT data for {self.name}.")

    def synchronise_pzt(self):
        pzt_time = self.pzt_start_times
        luna_time = self.luna_time_labels
        filecount = len(self.pzt_database)
        self.pzt_dt, best_error = sync_pzt(pzt_time, luna_time, self.ae_ribbons, filecount, name=self.name)
        print(f"PZT data should be shifted by {self.pzt_dt} seconds in order to achieve the best synchronization.")
        print(f"This synchronization gives an error of {best_error}.")

    def analyse_pzt(self):
        location = 'Files/' + self.name + "/PZT/" + self.name + "_PZT-clustered.csv"

        try:
            if self.force_clustering:
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

            # add state number
            time_list, state_list = list(big_df["time"].drop_duplicates()), list(range(1, len(set(big_df["time"])) + 1))
            time_list.sort()
            state_column = big_df["time"].rename({'time': 'state'}, axis=1).replace(time_list, state_list)
            big_df["state"] = state_column

            # reorder and sort big_df on time
            big_df['time'] = big_df['time'] - min(big_df['time'])
            big_df['time'] = big_df['time'] + self.pzt_dt
            self.pzt_clustered_database = big_df[['time', 'state', 'frequency', 'actionneur', 'max_amp', 'min_amp',
                                                 'avg_abs_amp', 'relative_amp', 'duration', 'rise_time', 'travel_time',
                                                 'energy', 'avg_freq']].sort_values(by=['time', "actionneur"])

            pd.DataFrame(self.pzt_clustered_database).to_csv(location, index=False)
            print("Successfully created PZT clustered .csv.")

        # call plotting function
        make_clusters(self.pzt_clustered_database, self.name)

        print(f"Successfully analysed PZT data for {self.name}.")

    def visualize_all(self):
        figure = plt.figure(tight_layout=True)
        figure.suptitle(f'Panel {self.name}')

        sub_figures = figure.subfigures(1, 1)

        # LUNA left foot.
        axs0 = sub_figures.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 5]})
        axs0[0].scatter(self.luna_database_filtered[0][:, 0], self.luna_database_filtered[0][:, 1],
                        c=self.luna_database_filtered[0][:, 2], cmap='bwr')
        axs0[0].set_ylabel('length [mm]')
        axs0[0].set_title('LUNA left foot cluster')

        # LUNA right foot.
        axs0[1].scatter(self.luna_database_filtered[1][:, 0], self.luna_database_filtered[1][:, 1],
                        c=self.luna_database_filtered[1][:, 2], cmap='bwr')
        axs0[1].set_ylabel('length [mm]')
        axs0[1].set_title('LUNA right foot cluster')

        axs0[2].scatter(self.ae_clustered_database['time'], self.ae_clustered_database['energy'],
                        c=self.ae_clustered_database['frequency_outlier'], cmap='bwr', s=4)
        axs0[2].set_xlabel('time [s]')
        axs0[2].set_ylabel('Energy [J]')
        axs0[2].set_title('AE cluster')

        plt.show()

    def save_all(self):
        """Function to save all relevant data to file"""
        directory = f'{self.folder_parent}/Files/{self.name}/Clusters'

        if not os.path.exists(directory):
            os.makedirs(directory)

        LUNA_data_to_save = np.vstack((self.luna_database_filtered[0], self.luna_database_filtered[1]))
        AE_data_to_save = pd.DataFrame(self.ae_clustered_database)

        with open(f'{directory}/LUNA.csv', 'w') as file:
            np.savetxt(file, LUNA_data_to_save, delimiter=',', fmt='%1.3f')

        with open(f'{directory}/AE.csv', 'w') as file:
            AE_data_to_save.to_csv(file, index=False)

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
