import os
# adjust to your liking
HOME_DIR = '/home/adrian'

class Config:
	def __init__(self, data_root=None):
		if data_root:
			self.data_root = data_root
		else:
			self.data_root = os.path.join(HOME_DIR, 'data', 'asc')
			
		self.moresque_dir = os.path.join(self.data_root, 'moresque')
		self.ambient_dir = os.path.join(self.data_root, 'ambient')
		self.odp239_dir = os.path.join(self.data_root, 'odp239')

