import vallenae as vae


class Pridb:
    def __init__(self, file_name):
        self.filename = file_name
        self.hits = None
    
    def get_hits(self):
        pridb = vae.io.PriDatabase("Files/"+self.filename+".pridb")
        hits = pridb.read_hits()
        self.hits = hits

    def save_csv(self):
        pass
