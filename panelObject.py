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
    def __init__(self, name, debug=False, force_clustering=False, force_saving=False, plotting=False):
        # General
        self.name = name
        self.debug = debug
        self.force_clustering = force_clustering
        self.force_saving = force_saving
        self.plotting = plotting

        # Results
        self.results_directory = 'Files/' + self.name + '/Results'
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        # AE
        self.ae_database = None
        self.ae_clustered_database = None
        self.ae_ribbons = None

        # LUNA
        self.luna_database = None
        self.luna_database_derivatives = None
        self.luna_database_clustered = None
        self.luna_database_filtered = None
        self.luna_database_visualize = None

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
    def initialize_all(debug=False, force_clustering=False, plotting=False):
        """A static method which checks the folders present and generates a Panel object for every folder"""
        if force_clustering:
            print("Force clustering is set to True, all datafiles will be regenerated...")
        entries = os.scandir(files_folder)
        lst = []

        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name, debug=debug, force_clustering=force_clustering))
        return lst

    # All the AE related code for the object
    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv(self.force_clustering)
        # print(self.ae_database.hits)
        print(f"Successfully loaded AE data for {self.name}...")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        # Try to find a clustered file else cluster the data
        location = 'Files/' + self.name + "/AE/" + self.name + "-clustered.csv"
        try:
            if self.force_clustering:
                raise FileNotFoundError
            self.ae_clustered_database = pd.read_csv(self.results_directory + "/AE.csv")
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
        """Function which takes all the internal variables related to the separate sensors and time synchronises them"""
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

    def visualize_luna(self):
        """
        This function prepares LUNA data to be plotted later.
        """
        left_positive, left_negative = [], []
        right_positive, right_negative = [], []

        for i in range(len(self.luna_database_filtered[0][:, 2])):
            if self.luna_database_filtered[0][i, 2] < 0:
                left_negative.append([self.luna_database_filtered[0][i, 0], self.luna_database_filtered[0][i, 1]])
            else:
                left_positive.append([self.luna_database_filtered[0][i, 0], self.luna_database_filtered[0][i, 1]])

        for i in range(len(self.luna_database_filtered[1][:, 2])):
            if self.luna_database_filtered[1][i, 2] < 0:
                right_negative.append([self.luna_database_filtered[1][i, 0], self.luna_database_filtered[1][i, 1]])
            else:
                right_positive.append([self.luna_database_filtered[1][i, 0], self.luna_database_filtered[1][i, 1]])

        left_positive, left_negative = np.array(left_positive), np.array(left_negative)
        right_positive, right_negative = np.array(right_positive), np.array(right_negative)

        self.luna_database_visualize = [left_positive, left_negative, right_positive, right_negative]

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
        self.pzt_dt, best_error = sync_pzt(pzt_time, luna_time, self.ae_ribbons, filecount, name=self.name, graphing=self.plotting)
        print(f"PZT data should be shifted by {self.pzt_dt} seconds in order to achieve the best synchronization.")
        print(f"This synchronization gives an error of {best_error}.")

    def analyse_pzt(self):
        try:
            if self.force_clustering:
                raise FileNotFoundError
            print(f"Successfully loaded clustered PZT data for {self.name}.")
            self.pzt_clustered_database = pd.read_csv(self.results_directory + "/PZT.csv")
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

            print("Successfully created PZT database.")

        # call plotting function
        make_clusters(self.pzt_clustered_database, self.name)

    def visualize_all(self):
        self.visualize_luna()

        figure = plt.figure(tight_layout=True)
        figure.suptitle(f'Panel {self.name}')

        sub_figures = figure.subfigures(1, 1)

        # LUNA left foot.
        axs0 = sub_figures.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 5]})

        if len(self.luna_database_visualize[0]) > 0:
            axs0[0].scatter(self.luna_database_visualize[0][:, 0], self.luna_database_visualize[0][:, 1],
                            color='red', label='Tension')
        if len(self.luna_database_visualize[1]) > 0:
            axs0[0].scatter(self.luna_database_visualize[1][:, 0], self.luna_database_visualize[1][:, 1],
                            color='blue', label='Compression')

        axs0[0].set_ylabel('length [mm]')
        axs0[0].set_title('LUNA left foot cluster')
        axs0[0].legend(loc='lower right')

        # LUNA right foot.
        if len(self.luna_database_visualize[2]) > 0:
            axs0[1].scatter(self.luna_database_visualize[2][:, 0], self.luna_database_visualize[2][:, 1],
                            color='red', label='Tension')

        if len(self.luna_database_visualize[3]) > 0:
            axs0[1].scatter(self.luna_database_visualize[3][:, 0], self.luna_database_visualize[3][:, 1],
                            color='blue', label='Compression')

        axs0[1].set_ylabel('length [mm]')
        axs0[1].set_title('LUNA right foot cluster')
        axs0[1].legend(loc='lower right')

        # AE cluster.
        axs0[2].scatter(self.ae_clustered_database['time'][self.ae_clustered_database['frequency_outlier'] == -1],
                        self.ae_clustered_database['frequency'][self.ae_clustered_database['frequency_outlier'] == -1],
                        c='red', s=4, label='frequency-amp outliers')
        axs0[2].scatter(self.ae_clustered_database['time'][self.ae_clustered_database['frequency_outlier'] == 0],
                        self.ae_clustered_database['frequency'][self.ae_clustered_database['frequency_outlier'] == 0],
                        c='blue', s=4, label='frequency-amp normal')
        axs0[2].legend()
        axs0[2].set_xlabel('time [s]')
        axs0[2].set_ylabel('Energy [J]')
        axs0[2].set_title('AE cluster')
        axs0[2].vlines(np.array(self.pzt_start_times)+self.pzt_dt-self.pzt_start_times[0], 50000, 300000, colors='g')

        plt.show()

    def save_all(self):
        """Function to save all relevant data to file"""
        directory = self.results_directory

        luna_data_to_save = np.vstack((self.luna_database_filtered[0], self.luna_database_filtered[1]))
        ae_data_to_save = self.ae_clustered_database
        pzt_data_to_save = self.pzt_clustered_database

        if not os.path.isfile(f'{directory}/LUNA.csv') or self.force_saving:
            with open(f'{directory}/LUNA.csv', 'w') as file:
                np.savetxt(file, luna_data_to_save, delimiter=',', fmt='%1.3f')
                print("Successfully created LUNA .csv.")

        if not os.path.isfile(f'{directory}/AE.csv') or self.force_saving:
            with open(f'{directory}/AE.csv', 'w') as file:
                ae_data_to_save.to_csv(file, index=False)
                print("Successfully created AE .csv.")

        if not os.path.isfile(f'{directory}/PZT.csv') or self.force_saving:
            with open(f'{directory}/PZT.csv', 'w') as file:
                pzt_data_to_save.to_csv(file, index=False)
                print("Successfully created PZT .csv.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
