import vallenae as vae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
        # uncomment for scatter correlation matrix (scatter + histogram)
        '''cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time','counts']
        df = pd.read_csv('Files/' + self.filename + "/AE/" + self.filename + ".csv", usecols=cols,
                                 delimiter=',').sample(n = 25000)
        grr = pd.plotting.scatter_matrix(df,figsize=(15,15), marker='o', s=60, alpha=.8)
        plt.show()'''

        # uncomment for correlation matrix (color coded)
        '''cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time', 'counts']
        data_array = pd.read_csv('Files/' + self.filename + "/AE/" + self.filename + ".csv", usecols=cols,
                                 delimiter=',')
        sns.set_theme(style="white")
        df = pd.DataFrame(data=data_array.values, columns=cols).sample(n = 25000)
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()'''

        # most recent correlation matrix using SNS to plot regression line and the regression coef
        def reg_coef(x, y, label=None, color=None, **kwargs):
            ax = plt.gca()
            r, p = pearsonr(x, y)
            ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.5), xycoords='axes fraction', ha='center')
            ax.set_axis_off()

        cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time', 'counts']
        data_array = pd.read_csv('Files/' + self.filename + "/AE/" + self.filename + ".csv", usecols=cols,
                                 delimiter=',').sample(n = 250000)
        g = sns.PairGrid(data_array, height=2)
        g.map_diag(sns.distplot)
        g.map_lower(sns.regplot)
        g.map_upper(reg_coef)
        plt.show()

        pass

    def __str__(self):
        return f"Pridb object for {self.filename}"

    def __repr__(self):
        return f"Pridb({self.filename})"
