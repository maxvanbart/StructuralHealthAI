import os


from AE.utilities import Pridb
from AE.clustering import init_clustering

files_folder = "Files"


class Panel:
    """An object which represents a panel"""
    def __init__(self, name, debug=False):
        self.name = name
        self.ae_database = None
        self.debug = debug

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
        print(f"Succesfully loaded AE data for {self.name}.")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        init_clustering(self.ae_database, debug=self.debug)
        # self.ae_database.corr_matrix()
        print(f"Succesfully analysed AE data for {self.name}.")

    # All the LUNA related code for the object
    def load_luna(self):
        """A function to load the LUNA data"""
        pass
        # print(f"Succesfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        pass
        # print(f"Succesfully analysed LUNA data for {self.name}.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
