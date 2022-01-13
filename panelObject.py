import os

from LUNA.luna_main import demo
from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster, energy_time_cluster
from AE.feature_extraction import frequency_extraction
from PZT.analyze_pzt import analyse_pzt
from PZT.load_pzt import StatePZT

import pandas as pd

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, debug=False, debug_graph=False):
        self.name = name
        self.debug = debug
        self.debug_graph = debug_graph

        # AE
        self.ae_database = None
        self.ae_clustered_database = None
        self.ae_start_time = None

        # LUNA
        self.luna_database = None

        # PZT
        self.pzt_database = None
        self.pzt_start_times = None
        self.pzt_final_result = None

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

            # adding extracted features and clusters
            print(f"Clustering completed for {self.name}, features and clusters being added to database...")
            self.ae_clustered_database["frequency"] = frequency_extraction(self.ae_clustered_database)
            self.ae_clustered_database["energy_outlier"] = energy_time_cluster(self.ae_clustered_database)
            self.ae_clustered_database["frequency_outlier"] = freq_amp_cluster(self.ae_clustered_database)

            # create new CSV
            pd.DataFrame(self.ae_clustered_database).to_csv(location, index=False)
            print("Successfully created clustered .csv.")

        # self.ae_database.corr_matrix()
        # freq_amp_cluster(self.ae_clustered_database, plotting=True)
        # energy_time_cluster(self.ae_clustered_database, plotting=True)
        # batch_fre_amp_clst(self.ae_clustered_database)

        print(f"Successfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        # LUNA code related to loading stuff should go here
        pass
        # print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        # LUNA code relating to analysis should go here
        demo(self.name)
        print(f"Successfully analysed LUNA data for {self.name}.")

    def load_pzt(self):
        self.pzt_database = StatePZT.initialize_pzt(self.name)
        time_list = []
        for identifier in self.pzt_database:
            time_list += [x.start_time for x in self.pzt_database[identifier]]
        time_list.sort()
        self.pzt_start_times = time_list
        print(self.pzt_start_times)
        print(f"Successfully loaded PZT data for {self.name}.")

    def analyse_pzt(self):
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
        self.pzt_final_result = big_df
        print(f"Successfully analysed PZT data for {self.name}.")

    def save_pzt(self):
        location = 'Files/' + self.name + "/result-" + self.name + "-pzt.csv"
        pd.DataFrame(self.pzt_final_result).to_csv(location, index=False)

    def time_synchronise(self):
        """Function which takes all the internal variables related to the seperate sensors and time synchronises them"""
        pass

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
