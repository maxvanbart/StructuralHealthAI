import vallenae as vae
import numpy as np
import pandas as pd


class Pridb:
    """A class for everything related to pridb/csv files"""
    def __init__(self, file_name):
        self.filename = file_name
        self.hits = None

    def save_csv(self):
        """A function to save the hits from the pridb to a csv file"""
        pd.DataFrame(self.hits).to_csv('Files/'+self.filename+".csv", index=False, header=False)

    def load_csv(self):
        """A function which tries to load the data from a csv file, otherwise it will generate it from the pridb file"""
        try:
            self.hits = pd.read_csv('Files/'+self.filename+".csv")
        except FileNotFoundError:
            print('File not found, generating from pridb file')
            self.pridb_read_hits()
            self.save_csv()

    def pridb_read_hits(self):
        """Function to retrieve the hits from the pridb file"""
        pridb = vae.io.PriDatabase("Files/"+self.filename+".pridb")
        hits = pridb.read_hits()
        self.hits = hits
