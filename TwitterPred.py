import subprocess
import os

if __name__ == '__main__':	
	if not os.path.exists('collection'):
		subprocess.call(['python', 'collection.py'])
		subprocess.call(['python', 'centrality.py'])
	subprocess.call(['python', 'extract.py'])
	subprocess.call(['python', 'centrality2.py'])
	subprocess.call(['python', 'prediction.py'])
