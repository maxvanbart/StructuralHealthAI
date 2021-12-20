import os


from AE.utilities import Pridb
from AE.hit_combination import init_clustering
from AE.feature_analysis import freq_amp_cluster, all_features_cluster, create_cluster_batches, energy_time_cluster, freq_amp_energy_plot
from AE.clustering import clustering_time_energy
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

        # self.ae_database.corr_matrix()
        # freq_amp_cluster(self.ae_clustered_database)
        # all_features_cluster(self.ae_clustered_database)
        # freq_amp_time_cluster(self.ae_clustered_database)
        # energy_time_cluster(self.ae_clustered_database)
        # freq_amp_energy_plot(self.ae_database, title="Frequency, amplitude and energy for uncombined randomly sampled emissions in the L1-03 panel")

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
        pass
        # print(f"Successfully analysed LUNA data for {self.name}.")

    def time_synchronise(self):
        """Function which takes all the internal variables related to the seperate sensors and time synchronises them"""
        pass

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
