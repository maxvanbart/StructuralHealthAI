import os

# from AE.utilities import Pridb
# from AE.hit_combination import init_clustering
# from AE.feature_analysis import freq_amp_cluster

from LUNA.luna_data_to_array import folder_to_array, gradient_arrays, array_to_image
from LUNA.luna_array_to_cluster import array_to_cluster, cluster_to_image
from LUNA.luna_plotting import plot_cluster

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

    def plot_ae(self):
        pass

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        luna_data_left, luna_data_right = folder_to_array(self.name, self.folder_luna)
        luna_data_left_time, luna_data_left_length = gradient_arrays(luna_data_left)
        luna_data_right_time, luna_data_right_length = gradient_arrays(luna_data_right)

        self.luna_database = [luna_data_left_time, luna_data_right_time, luna_data_left_length, luna_data_right_length]

        print(f"Successfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        time_left, time_right, length_left, length_right = self.luna_database

        self.luna_database_clustered = array_to_cluster(time_left, time_right, length_left, length_right)

        print(f"Successfully analysed LUNA data for {self.name}.")

    def plot_luna(self):
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

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"


panel = Panel('L1-23')
panel.load_luna()
panel.analyse_luna()
panel.plot_luna()
