import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import energy_time_cluster, freq_amp_cluster, AE_plot_visualisation
from AE.feature_extraction import frequency_extraction

from PZT.analyze_pzt import analyse_pzt, make_clusters, extract_scores
from PZT.load_pzt import StatePZT

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays
from LUNA.luna_array_to_cluster import array_to_cluster
from LUNA.luna_preprocessing import preprocess_array
from LUNA.luna_postprocessing import filter_array

from TimeSync.timeSync import sync_luna, sync_pzt

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, pzt_thr, debug=False, force_clustering=False, plotting=False):
        # General
        self.name = name
        self.debug = debug
        self.force_clustering = force_clustering
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
        self.pzt_threshold = pzt_thr

    @staticmethod
    def initialize_all(pzt_thr, debug=False, force_clustering=False, plotting=False):
        """A static method which checks the folders present and generates a Panel object for every folder"""
        if force_clustering:
            print("Force clustering is set to True, all datafiles will be regenerated...")
        entries = os.scandir(files_folder)
        lst = []
        for entry in entries:
            if entry.is_dir():
                lst.append(Panel(entry.name, pzt_thr, debug=debug, force_clustering=force_clustering, plotting=plotting))
        return lst

    # All the AE related code for the object
    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv(force_clustering=self.force_clustering)
        print(f"Successfully loaded AE data for {self.name}.")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        # Try to find a clustered file else cluster the data
        location = 'Files/' + self.name + "/AE/" + self.name + "-clustered.csv"
        try:
            if self.force_clustering:
                raise FileNotFoundError
            self.ae_clustered_database = pd.read_csv(self.results_directory + f"/AE_{self.name}.csv")
            print(f"Successfully loaded clustered AE data for {self.name}.")
        except FileNotFoundError:
            print('Clustered file not found, clustering data...')
            # creation of clustered database
            self.ae_clustered_database = self.ae_database.hits
            # detection of energy outliers
            self.ae_clustered_database["energy_outlier"] = energy_time_cluster(self.ae_clustered_database,
                                                                               self.results_directory,
                                                                               self.name)
            # removal of the energy outlier
            self.ae_clustered_database = self.ae_clustered_database[self.ae_clustered_database["energy_outlier"] == 1]
            # hits combination
            self.ae_clustered_database = init_clustering(self.ae_clustered_database, debug=self.debug)
            # adding frequency to the database
            self.ae_clustered_database["frequency"] = frequency_extraction(self.ae_clustered_database)
            # frequency outlier detection
            self.ae_clustered_database["frequency_outlier"] = freq_amp_cluster(self.ae_clustered_database,
                                                                               self.results_directory,
                                                                               self.name)
            # adding extracted features and clusters
            print(f"Clustering completed for {self.name}, features and clusters being added to database...")
            print(f"Successfully analysed AE data for {self.name}.")

        self.ae_clustered_database = self.ae_clustered_database.sort_values(by=['time'])

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

        print(f"Successfully loaded LUNA data for {self.name}.")

    def synchronise_luna(self):
        """Function which takes all the internal variables related to the separate sensors and time synchronises them"""
        if not os.path.isfile(f'{self.results_directory}/LUNA_left_{self.name}.csv') or self.force_clustering:
            sv, e, rb = sync_luna(self.ae_database.hits, self.luna_file_vector, self.luna_time_labels, name=self.name)
            self.luna_time_shift_vector = sv
            self.luna_time_shift_errors = e

            self.luna_time_labels = self.luna_time_labels + self.luna_time_shift_vector
            self.ae_ribbons = rb

            print(f"Successfully time-synchronized LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""

        if not os.path.isfile(f'{self.results_directory}/LUNA_left_{self.name}.csv') or self.force_clustering:
            # 1. get time and length derivatives.
            left_time, left_length = gradient_arrays(self.luna_database[0])
            right_time, right_length = gradient_arrays(self.luna_database[1])

            # 2. get clustered database.
            self.luna_database_derivatives = [left_time, right_time, left_length, right_length]
            self.luna_database_clustered = array_to_cluster(left_time, right_time, left_length, right_length)

            # 3. filter original database with clustered database.
            left_filtered = filter_array(self.luna_database[0], self.luna_database_clustered[0], self.luna_time_labels, self.luna_length_labels[0])
            right_filtered = filter_array(self.luna_database[1], self.luna_database_clustered[1], self.luna_time_labels, self.luna_length_labels[1])

        else:
            with open(f'{self.results_directory}/LUNA_left_{self.name}.csv') as file:
                left_filtered = np.genfromtxt(file, delimiter=',')
            with open(f'{self.results_directory}/LUNA_right_{self.name}.csv') as file:
                right_filtered = np.genfromtxt(file, delimiter=',')

        self.luna_database_filtered = [left_filtered, right_filtered]

        print(f"Successfully analysed LUNA data for {self.name}.")

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
        self.pzt_database = StatePZT.initialize_pzt(self.name, self)
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
        self.pzt_dt, best_error = sync_pzt(pzt_time, luna_time, self.ae_ribbons, filecount, self.results_directory,
                                           name=self.name, graphing=self.plotting)
        print(f"Successfully time-synchronized PZT data for {self.name}.")

    def analyse_pzt(self):
        try:
            if self.force_clustering:
                raise FileNotFoundError
            print(f"Successfully loaded clustered PZT data for {self.name}.")
            self.pzt_clustered_database = pd.read_csv(self.results_directory + f"/PZT_{self.name}.csv")
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
        make_clusters(self.pzt_clustered_database, self.name, self.results_directory, self.plotting)
        print(f"Successfully analysed PZT data for {self.name}.")

    def visualize_all(self, plotting):
        # plot AE visualisations
        AE_plot_visualisation(self.ae_clustered_database, self.results_directory, self.name, plotting=plotting)

        # plot combined visualisations
        self.visualize_luna()

        figure = plt.figure(constrained_layout=True, figsize=(12, 9))

        sub_figures = figure.subfigures(1, 1)
        sub_figures.suptitle(f'Panel {self.name}')
        # LUNA left foot.


        # Open PZT states of interest
        # extract_scores(self.results_directory, self.name)

        axs0 = sub_figures.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})

        if len(self.luna_database_visualize[0]) > 0:
            axs0[0].scatter(self.luna_database_visualize[0][:, 0], self.luna_database_visualize[0][:, 1],
                            color='tab:red', label='Tension')
        if len(self.luna_database_visualize[1]) > 0:
            axs0[0].scatter(self.luna_database_visualize[1][:, 0], self.luna_database_visualize[1][:, 1],
                            color='tab:blue', label='Compression')

        axs0[0].set_ylabel('Length [mm]')
        axs0[0].set_title('LUNA left foot cluster')
        axs0[0].legend(loc='lower right')

        # LUNA right foot.
        if len(self.luna_database_visualize[2]) > 0:
            axs0[1].scatter(self.luna_database_visualize[2][:, 0], self.luna_database_visualize[2][:, 1],
                            color='tab:red', label='Tension')

        if len(self.luna_database_visualize[3]) > 0:
            axs0[1].scatter(self.luna_database_visualize[3][:, 0], self.luna_database_visualize[3][:, 1],
                            color='tab:blue', label='Compression')

        axs0[1].set_xlabel("Time [s]")
        axs0[1].set_ylabel('Length [mm]')
        axs0[1].set_title('LUNA right foot cluster')
        axs0[1].legend(loc='lower right')

        # AE energy plot.
        axs0[2].scatter(self.ae_clustered_database['time'],
                        self.ae_clustered_database['energy'],
                        s=10, label='High energy events', c="tab:blue")
        axs0[2].set_xlabel("Time [s]")
        axs0[2].set_ylabel("Peak energy of emission [$10^{-14}$ J]")
        axs0[2].set_title('AE energy plot')
        axs0[2].vlines(np.array(self.pzt_start_times) + self.pzt_dt - self.pzt_start_times[0],
                       ymin=min(self.ae_clustered_database['energy']), ymax=max(self.ae_clustered_database['energy']),
                       colors='tab:orange', label='PZT measurements')
        axs0[2].legend()

        plt.savefig(f'{self.results_directory}/combined_LUNA-PZT-AE energy_{self.name}.png',  dpi=200)
        if plotting:
            plt.show()

        # Separate AE plot for clustered frequency and amplitude
        plt.figure(figsize=(11, 7))
        plt.scatter(self.ae_clustered_database['time'][self.ae_clustered_database['frequency_outlier'] == -1],
                    self.ae_clustered_database['frequency'][self.ae_clustered_database['frequency_outlier'] == -1],
                    s=3, c='#334451', label='AE frequency outliers')
        plt.scatter(self.ae_clustered_database['time'][self.ae_clustered_database['frequency_outlier'] == 0],
                    self.ae_clustered_database['frequency'][self.ae_clustered_database['frequency_outlier'] == 0],
                    s=3, c='tab:blue', label='AE non-outliers')

        plt.vlines(np.array(self.pzt_start_times) + self.pzt_dt - self.pzt_start_times[0],
                   ymin=min(self.ae_clustered_database['frequency']), ymax=max(self.ae_clustered_database['frequency']),
                   colors='tab:orange', label='PZT measurements')
        plt.xlabel("Time [s]")
        plt.ylabel("Average frequency of emission [kHz]")
        plt.legend()
        plt.title(f'Panel {self.name}: clustered AE emissions shown with timestamps of PZT measurements')

        plt.savefig(f'{self.results_directory}/combined_PZT-AE freq_{self.name}.png', dpi=200)
        if plotting:
            plt.show()

    def save_all(self):
        """Function to save all relevant data to file"""
        directory = self.results_directory

        luna_data_to_save_left = self.luna_database_filtered[0]
        luna_data_to_save_right = self.luna_database_filtered[1]

        ae_data_to_save = self.ae_clustered_database
        pzt_data_to_save = self.pzt_clustered_database

        if not os.path.isfile(f'{directory}/LUNA_left_{self.name}.csv') or not os.path.isfile(f'{directory}/LUNA_right_{self.name}.csv') \
                or self.force_clustering:
            with open(f'{directory}/LUNA_left_{self.name}.csv', 'w') as file:
                np.savetxt(file, luna_data_to_save_left, delimiter=',', fmt='%1.3f')
            with open(f'{directory}/LUNA_right_{self.name}.csv', 'w') as file:
                np.savetxt(file, luna_data_to_save_right, delimiter=',', fmt='%1.3f')
                print("Successfully created LUNA .csv.")

        if not os.path.isfile(f'{directory}/AE_{self.name}.csv') or self.force_clustering:
            with open(f'{directory}/AE_{self.name}.csv', 'w') as file:
                ae_data_to_save.to_csv(file, index=False)
                print("Successfully created AE .csv.")

        if not os.path.isfile(f'{directory}/PZT_{self.name}.csv') or self.force_clustering:
            with open(f'{directory}/PZT_{self.name}.csv', 'w') as file:
                pzt_data_to_save.to_csv(file, index=False)
                print("Successfully created PZT .csv.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
