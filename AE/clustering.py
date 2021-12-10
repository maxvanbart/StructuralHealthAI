from AE.utilities import Pridb

def init_clustering():
    database = Pridb("L1-03")
    database.load_csv()
    features = database.hits
    print(features.columns)
