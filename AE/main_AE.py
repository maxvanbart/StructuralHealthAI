import numpy as np
import vallenae as vae

from AE.utilities import Pridb

def main():
    print('well hello there')

    database = Pridb("L1-03.pridb")
    hits = database.return_hits()
