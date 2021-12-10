import numpy as np
import vallenae as vae

from AE.utilities import Pridb


def analysis_ae():
    database = Pridb("L1-03")
    database.load_csv()
    print(database.hits)
