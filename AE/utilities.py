import vallenae as vae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import datetime


class Pridb:
    """A class for everything related to pridb/csv files"""
    def __init__(self, file_name):
        self.filename = file_name
        self.hits = None
        self.abs_start_time = None

        # Make a list of all pridb files
        content = os.listdir(f"Files/{self.filename}/AE")
        self.pridb_files = []
        for item in content:
            if '.pridb' in item:
                self.pridb_files.append(item)

    def save_csv(self):
        """A function to save the hits from the pridb to a csv file"""
        pd.DataFrame(self.hits).to_csv('Files/'+self.filename+"/AE/"+self.filename+".csv", index=False)

    def load_csv(self, force_clustering):
        """A function which tries to load the data from a csv file, otherwise it will generate it from the pridb file"""
        try:
            if force_clustering:
                raise FileNotFoundError
            self.hits = pd.read_csv('Files/'+self.filename+"/AE/"+self.filename+".csv")
        except FileNotFoundError:
            print('File not found, generating from pridb file')
            self.pridb_read_hits()
            self.save_csv()

    def pridb_read_hits(self):
        """Function to retrieve the hits from the pridb file"""
        start_time_dict = {}
        hits_dict = {}
        # We first open all pridb files which where found during init
        for file in self.pridb_files:
            pridb = vae.io.PriDatabase("Files/" + self.filename + "/AE/" + file)
            # Extract the start time for the pridb file
            markers = pridb.read_markers()
            start_time_dict[file] = markers["data"].loc[3]
            # Extract the hits for the pridb file
            hits_dict[file] = pridb.read_hits()

        # Here we convert the starting times for all files to a datetime timestamp format
        start_time_dict = {x: convert_to_datetime(start_time_dict[x]) for x in start_time_dict}
        # We use this dictionary to determine the earliest starting time
        self.abs_start_time = min(start_time_dict.values())

        # From the earliest starting time we determine the absolute time delta between files
        delta_time_dict = {x: start_time_dict[x] - self.abs_start_time for x in start_time_dict}

        # We now add the time delta to every dataframe such that they can be appended to each other
        for file in hits_dict:
            dt = delta_time_dict[file]
            df = hits_dict[file]
            df['time'] = df['time'].map(lambda time: time+dt)
            hits_dict[file] = df

        # Here we append all the different dataframes for the panel together
        final_df = None
        for file in hits_dict:
            if final_df is None:
                final_df = hits_dict[file]
            else:
                final_df = pd.concat([final_df, hits_dict[file]], axis=0)

        # We finally add the absolute time for every measurement and return the final dataframe
        final_df['abs_time'] = final_df['time'] + self.abs_start_time
        final_df = final_df.sort_values(by=['time'])
        self.hits = final_df

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

        # most recent correlation matrix using SNS to plot distribution function
        # regression line and the regression coef
        def reg_coef(x, y, label=None, color=None, **kwargs):
            ax = plt.gca()
            r, p = pearsonr(x, y)
            ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.5), xycoords='axes fraction', ha='center')
            ax.set_axis_off()

        cols = ['time', 'amplitude', 'duration', 'energy', 'rms', 'rise_time', 'counts']
        data_array = pd.read_csv('Files/' + self.filename + "/AE/" + self.filename + ".csv", usecols=cols,
                                 delimiter=',').sample(n=25000)
        g = sns.PairGrid(data_array, height=2)
        g.map_diag(sns.distplot)
        g.map_lower(sns.regplot)
        g.map_upper(reg_coef)
        plt.show()

    def __str__(self):
        return f"Pridb object for {self.filename}"

    def __repr__(self):
        return f"Pridb({self.filename})"


def convert_to_datetime(text):
    """Function to convert a datetime string to a timestamp"""
    date, t = text.split(' ')
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)

    hour, minute, seconds = t.split(':')
    hour, minute, seconds = int(hour), int(minute), int(seconds)
    second = int(seconds // 1)
    milisecond = int(round((seconds % 1)*10**6))
    date_obj = datetime.datetime(year, month, day, hour, minute, second, milisecond)
    return datetime.datetime.timestamp(date_obj)
