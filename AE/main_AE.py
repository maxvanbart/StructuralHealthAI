import numpy as np
import vallenae as vae

from AE.utilities import Pridb


def main():
    database = Pridb("L1-03.pridb")
    database.return_hits()
    print(database.hits)
