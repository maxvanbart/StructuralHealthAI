import numpy as np
import vallenae as vae

from AE.utilities import Pridb


# this entire file is completely useless but could be used for testing of a single panel
def analysis_ae():
    database = Pridb("L1-03")
    database.load_csv()
    print(database.hits)
