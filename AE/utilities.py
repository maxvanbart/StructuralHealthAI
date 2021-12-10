import vallenae as vae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Pridb:
    """A class for everything related to pridb/csv files"""
    def __init__(self, file_name):
        self.filename = file_name
        self.hits = None

    def save_csv(self):
        """A function to save the hits from the pridb to a csv file"""
        pd.DataFrame(self.hits).to_csv('Files/'+self.filename+"/AE/"+self.filename+".csv", index=False)

    def load_csv(self):
        """A function which tries to load the data from a csv file, otherwise it will generate it from the pridb file"""
        try:
            self.hits = pd.read_csv('Files/'+self.filename+"/AE/"+self.filename+".csv")
        except FileNotFoundError:
            print('File not found, generating from pridb file')
            self.pridb_read_hits()
            self.save_csv()

    def pridb_read_hits(self):
        """Function to retrieve the hits from the pridb file"""
        pridb = vae.io.PriDatabase("Files/"+self.filename+"/AE/"+self.filename+".pridb")
        hits = pridb.read_hits()
        self.hits = hits

    def corr_matrix(self):
        """Creation of correlation matrix for a panel using CSV file"""
        data_array = pd.read_csv('Files/' + self.filename + "/AE/" + self.filename + ".csv",
                                 delimiter=',').values
        sns.set_theme(style="white")
        df = pd.DataFrame(data=data_array, columns=self.hits.columns)
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    def __str__(self):
        return f"Pridb object for {self.filename}"

    def __repr__(self):
        return f"Pridb({self.filename})"
