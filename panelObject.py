import os


from AE.utilities import Pridb


files_folder = "Files"


class Panel:
    def __init__(self, name):
        self.name = name
        self.ae_database = None

    @staticmethod
    def initialize_all():
        """A static method which checks the folders present and generates a Panel object for every folder"""
        panels = os.listdir(files_folder)
        lst = []
        for panel in panels:
            lst.append(Panel(panel))

        return lst

    def load_ae(self):
        """Function to load the AE data in the folder"""
        self.ae_database = Pridb(self.name)
        self.ae_database.load_csv()
        print(self.ae_database.hits)
        print("Succesfully loaded AE data for {self.name}.")

    def analyse_ae(self):
        """Function to analyse the AE data in the folder"""
        pass
        # print("Succesfully analysed AE data for {self.name}.")

    def load_luna(self):
        """A function to load the LUNA data"""
        pass
        # print("Succesfully loaded LUNA data for {self.name}.")

    def analyse_luna(self):
        """A function to analyse the LUNA data in the folder"""
        pass
        # print("Succesfully analysed LUNA data for {self.name}.")

    def __repr__(self):
        return f"PanelObject({self.name})"

    def __str__(self):
        return f"Panel {self.name}"
