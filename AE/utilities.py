import vallenae as vae


class Pridb():
  def __init__(self):
    self.id = 1
    
  def return_hits(self, file_name):
    pridb = vae.io.PriDatabase(file_name)
    hits = pridb.read_hits()
    return hits