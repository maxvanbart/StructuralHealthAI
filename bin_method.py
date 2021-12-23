import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from LUNA.luna_data_to_array import file_to_array
from TimeSync.timeSync import sync_time


# Name of the panel to test on
name = 'L1-09'

# Load AE dataframe
location_ae = 'Files/' + name + "/AE/" + name + "-clustered.csv"
ae_clustered_database = pd.read_csv(location_ae)
ae_df = ae_clustered_database.sort_values(by=['time'])

# Load LUNA data
location_luna = f'Files/{name}/LUNA/{name}-FTest.txt'
array_luna, _, _, _ = file_to_array(name, location_luna)
print(type(array_luna))

sync_time(ae_df, array_luna, name=name)
