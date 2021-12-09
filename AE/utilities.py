import vallenae as vae


class Pridb():
  def __init__(self, file_name):
    self.filename = file_name
    
  def return_hits(self):
    pridb = vae.io.PriDatabase(self.filename)
    hits = pridb.read_hits()
    return hits